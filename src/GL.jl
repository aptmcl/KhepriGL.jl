export gl

#=
KhepriGL — Fast OpenGL backend for Khepri

Architecture: accumulate all geometry into flat buffers, upload to GPU in a
single batch at render time. We implement only the lowest-level primitives
(b_trig, b_quad, b_ngon, b_quad_strip, b_line, b_point); KhepriBase's
defaults decompose all higher-level shapes down to these.
=#

# ─── Backend type definitions ─────────────────────────────────────────────────

abstract type GLKey end
const GLId = Int
const GLIds = Vector{GLId}
const GLRef = GenericRef{GLKey, GLId}
const GLRefs = Vector{GLRef}
const GLNativeRef = NativeRef{GLKey, GLId}

# ─── Scene buffers (chunked, interleaved) ────────────────────────────────────
# Interleaved vertex format: pos(3) + norm(3) + col(4) = 10 floats per vertex.
# Fixed-size CPU buffers are filled, then flushed as GPU chunks (VBO+VAO).

const VERTEX_FLOATS = 10  # pos(3) + norm(3) + col(4) per vertex
const CHUNK_MAX_VERTICES = 65_532  # ~2.5 MB per chunk, divisible by 2,3,4,6,12

struct GPUChunk
  vao::UInt32
  vbo::UInt32
  vertex_count::Int
  data::Vector{Float32}  # retained CPU copy for context loss (MSAA restart)
end

mutable struct PrimitiveBuffers
  chunks::Vector{GPUChunk}                       # completed chunks on GPU
  to_delete::Vector{GPUChunk}                     # chunks awaiting GPU resource cleanup
  pending::Vector{Tuple{Vector{Float32}, Int}}    # (data, vertex_count) awaiting GPU upload
  buffer::Vector{Float32}                         # current fixed-size CPU buffer (reused)
  offset::Int                                     # write position in floats
  current_vao::UInt32                             # VAO for partial buffer (reused)
  current_vbo::UInt32                             # VBO for partial buffer (reused)
  dirty::Bool
end

PrimitiveBuffers() = PrimitiveBuffers(
  GPUChunk[], GPUChunk[],
  Tuple{Vector{Float32}, Int}[],
  Vector{Float32}(undef, CHUNK_MAX_VERTICES * VERTEX_FLOATS),
  0, UInt32(0), UInt32(0), false)

mutable struct GLScene
  tris::PrimitiveBuffers
  lines::PrimitiveBuffers
  points::PrimitiveBuffers
end

GLScene() = GLScene(PrimitiveBuffers(), PrimitiveBuffers(), PrimitiveBuffers())

function flush_buffer!(pb::PrimitiveBuffers)
  pb.offset == 0 && return
  nv = pb.offset ÷ VERTEX_FLOATS
  data = pb.buffer[1:pb.offset]  # copy used portion
  push!(pb.pending, (data, nv))
  pb.offset = 0
  pb.dirty = true
end

has_data(pb::PrimitiveBuffers) =
  !isempty(pb.chunks) || !isempty(pb.pending) || pb.offset > 0

function clear_primitive_buffers!(pb::PrimitiveBuffers)
  append!(pb.to_delete, pb.chunks)
  empty!(pb.chunks)
  empty!(pb.pending)
  pb.offset = 0
  pb.dirty = true
end

function clear_scene!(s::GLScene)
  clear_primitive_buffers!(s.tris)
  clear_primitive_buffers!(s.lines)
  clear_primitive_buffers!(s.points)
  s
end

# ─── Backend struct ───────────────────────────────────────────────────────────

@kwdef mutable struct GLBackend <: Backend{GLKey, GLId}
  shapes::Shapes = Shape[]
  current_layer::Union{Nothing, AbstractLayer} = nothing
  layers::Dict{AbstractLayer, Vector{Shape}} = Dict{AbstractLayer, Vector{Shape}}()
  date::DateTime = DateTime(2020, 9, 21, 10, 0, 0)
  place::GeographicLocation = GeographicLocation(39, 9, 0, 0)
  render_env::RenderEnvironment = RealisticSkyEnvironment(5, true)
  ground_level::Float64 = 0.0
  ground_material::Union{Nothing, Material} = nothing
  view::View = default_view()
  transaction::Parameter{KhepriBase.Transaction} = Parameter{KhepriBase.Transaction}(KhepriBase.AutoCommitTransaction())
  refs::References{GLKey, GLId} = References{GLKey, GLId}()
  scene::GLScene = GLScene()
  window::Any = nothing  # GLFW.Window or nothing

  # GPU handles (initialized on first render)
  # Per-primitive VAO/VBO resources are in PrimitiveBuffers (scene.tris/lines/points)
  shader_program::UInt32 = 0
  flat_shader_program::UInt32 = 0

  width::Int = 1024
  height::Int = 768
  initialized::Bool = false

  # Next reference ID (monotonically increasing)
  next_id::Int = 1

  # Interactive camera state
  orbit_active::Bool = false
  pan_active::Bool = false
  last_mouse_x::Float64 = 0.0
  last_mouse_y::Float64 = 0.0
  cam_distance::Float64 = 30.0
  cam_azimuth::Float64 = π/4
  cam_elevation::Float64 = π/6
  cam_target::Vector{Float64} = [0.0, 0.0, 0.0]

  # Background render timer for interactive mode
  render_timer::Any = nothing  # Timer or nothing

  # Configurable background color (RGBA)
  background_color::NTuple{4,Float32} = (0.9f0, 0.9f0, 0.92f0, 1.0f0)

  # Display mode: :shaded, :wireframe, :arctic, :xray, :shaded_wireframe
  display_mode::Symbol = :shaded

  # GPU handles for additional shaders
  arctic_shader_program::UInt32 = 0
  xray_shader_program::UInt32 = 0

  # MSAA anti-aliasing (0 = off, 2/4/8 = sample count)
  msaa_samples::Int = 4

  # Flag to trigger window recreation (e.g. after MSAA change)
  needs_restart::Bool = false

  # Last render_size applied to the window (to detect changes)
  applied_render_width::Int = 0
  applied_render_height::Int = 0
end

const GL = GLBackend

KhepriBase.backend_name(b::GL) = "GL"
KhepriBase.void_ref(b::GL) = GLNativeRef(0)

# Use frontend view management (stored in b.view)
KhepriBase.view_type(::Type{GL}) = FrontendView()

const gl = GL()

# ─── Shaders ──────────────────────────────────────────────────────────────────

const VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec4 vColor;

void main() {
  vec4 worldPos = uModel * vec4(aPos, 1.0);
  vWorldPos = worldPos.xyz;
  vWorldNormal = mat3(transpose(inverse(uModel))) * aNormal;
  vColor = aColor;
  gl_Position = uProjection * uView * worldPos;
}
"""

const FRAG_SHADER_PHONG_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec4 vColor;

uniform vec3 uCameraPos;

out vec4 FragColor;

void main() {
  vec3 N = normalize(vWorldNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);

  // Hemisphere ambient: upper hemisphere brighter than lower
  float hemisphereBlend = 0.5 + 0.5 * N.z;  // Z-up
  float ambient = mix(0.08, 0.20, hemisphereBlend);

  // Key light: headlight from camera direction
  vec3 L1 = V;
  vec3 H1 = normalize(L1 + V);
  float diff1 = max(dot(N, L1), 0.0) * 0.55;
  float spec1 = pow(max(dot(N, H1), 0.0), 32.0) * 0.25;

  // Fill light: fixed direction from below-left
  vec3 L2 = normalize(vec3(-0.4, -0.3, -0.5));
  float diff2 = max(dot(N, L2), 0.0) * 0.20;

  // Two-sided lighting for key light
  float diffBack = max(dot(-N, L1), 0.0) * 0.35;

  float lighting = ambient + max(diff1, diffBack) + diff2 + spec1;
  FragColor = vec4(vColor.rgb * lighting, vColor.a);
}
"""

const FRAG_SHADER_FLAT_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec4 vColor;

out vec4 FragColor;

void main() {
  FragColor = vColor;
}
"""

const FRAG_SHADER_ARCTIC_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec4 vColor;

uniform vec3 uCameraPos;

out vec4 FragColor;

void main() {
  vec3 N = normalize(vWorldNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);

  // Multi-directional soft lighting (3 lights)
  float d1 = max(dot(N, V), 0.0) * 0.45;                                // headlight
  float d2 = max(dot(N, normalize(vec3(0.5, 0.3, 0.8))), 0.0) * 0.35;   // top-right
  float d3 = max(dot(N, normalize(vec3(-0.3, -0.5, 0.4))), 0.0) * 0.20;  // fill

  // Fresnel edge darkening (edges appear slightly darker)
  float fresnel = 1.0 - max(dot(N, V), 0.0);
  float edgeDarken = 1.0 - fresnel * fresnel * 0.3;

  // Two-sided
  float d1back = max(dot(-N, V), 0.0) * 0.3;
  float lighting = 0.25 + max(d1, d1back) + d2 + d3;

  vec3 baseColor = vec3(0.95);  // near-white, ignore vertex colors
  FragColor = vec4(baseColor * lighting * edgeDarken, 1.0);
}
"""

const FRAG_SHADER_XRAY_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec4 vColor;

uniform vec3 uCameraPos;

out vec4 FragColor;

void main() {
  vec3 N = normalize(vWorldNormal);
  vec3 V = normalize(uCameraPos - vWorldPos);
  float edge = 1.0 - abs(dot(N, V));  // brighter at edges (silhouette)
  float alpha = 0.15 + edge * 0.4;
  FragColor = vec4(vColor.rgb * 0.7 + 0.3, alpha);
}
"""

# ─── Shader compilation ──────────────────────────────────────────────────────

function compile_shader(source, shader_type)
  shader = glCreateShader(shader_type)
  glShaderSource(shader, 1, Ptr{GLchar}[pointer(source)], C_NULL)
  glCompileShader(shader)
  success = Ref{GLint}(0)
  glGetShaderiv(shader, GL_COMPILE_STATUS, success)
  if success[] == GL_FALSE
    len = Ref{GLint}(0)
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, len)
    log = Vector{UInt8}(undef, len[])
    actual_len = Ref{GLsizei}(0)
    glGetShaderInfoLog(shader, len[], actual_len, log)
    error("Shader compilation failed: $(unsafe_string(pointer(log), actual_len[]))")
  end
  shader
end

function link_program(vert_shader, frag_shader)
  program = glCreateProgram()
  glAttachShader(program, vert_shader)
  glAttachShader(program, frag_shader)
  glLinkProgram(program)
  success = Ref{GLint}(0)
  glGetProgramiv(program, GL_LINK_STATUS, success)
  if success[] == GL_FALSE
    len = Ref{GLint}(0)
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, len)
    log = Vector{UInt8}(undef, len[])
    actual_len = Ref{GLsizei}(0)
    glGetProgramInfoLog(program, len[], actual_len, log)
    error("Shader link failed: $(unsafe_string(pointer(log), actual_len[]))")
  end
  glDeleteShader(vert_shader)
  glDeleteShader(frag_shader)
  program
end

function create_shader_program(vert_src, frag_src)
  vert = compile_shader(vert_src, GL_VERTEX_SHADER)
  frag = compile_shader(frag_src, GL_FRAGMENT_SHADER)
  link_program(vert, frag)
end

# ─── Window and OpenGL initialization ─────────────────────────────────────────

raise_window(win) =
  begin
    GLFW.SetWindowAttrib(win, GLFW.FLOATING, true)
    GLFW.SetWindowAttrib(win, GLFW.FLOATING, false)
  end

function ensure_window(b::GL)
  if b.window !== nothing && !Bool(GLFW.WindowShouldClose(b.window))
    return b.window
  end
  if b.window !== nothing
    GLFW.DestroyWindow(b.window)
    b.window = nothing
    b.initialized = false
  end
  GLFW.Init()
  GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
  GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
  GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
  GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, true)
  GLFW.WindowHint(GLFW.VISIBLE, false)
  GLFW.WindowHint(GLFW.SAMPLES, b.msaa_samples)
  b.width = render_width()
  b.height = render_height()
  b.window = GLFW.CreateWindow(b.width, b.height, "KhepriGL")
  GLFW.MakeContextCurrent(b.window)
  setup_callbacks(b)
  b.initialized = false
  b.window
end

function ensure_gpu(b::GL)
  b.initialized && return
  GLFW.MakeContextCurrent(b.window)

  glEnable(GL_DEPTH_TEST)
  glEnable(GL_BLEND)
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
  glDisable(GL_CULL_FACE)
  if b.msaa_samples > 0
    glEnable(GL_MULTISAMPLE)
  end
  glPointSize(5.0)
  glLineWidth(1.5)
  glClearColor(b.background_color...)

  # Compile shaders
  b.shader_program = create_shader_program(VERTEX_SHADER_SRC, FRAG_SHADER_PHONG_SRC)
  b.flat_shader_program = create_shader_program(VERTEX_SHADER_SRC, FRAG_SHADER_FLAT_SRC)
  b.arctic_shader_program = create_shader_program(VERTEX_SHADER_SRC, FRAG_SHADER_ARCTIC_SRC)
  b.xray_shader_program = create_shader_program(VERTEX_SHADER_SRC, FRAG_SHADER_XRAY_SRC)

  # Initialize GPU resources for chunked buffers
  for pb in (b.scene.tris, b.scene.lines, b.scene.points)
    rebuild_gpu!(pb)
  end

  b.initialized = true
end

function rebuild_gpu!(pb::PrimitiveBuffers)
  # Discard stale deletion queue (old context is gone)
  empty!(pb.to_delete)
  # Recreate current VAO/VBO
  pb.current_vao = create_vao()
  pb.current_vbo = create_vbo()
  setup_interleaved_vao!(pb.current_vao, pb.current_vbo)
  # Rebuild existing chunks from retained CPU data
  for i in 1:length(pb.chunks)
    chunk = pb.chunks[i]
    vao = create_vao()
    vbo = create_vbo()
    setup_interleaved_vao!(vao, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(chunk.data), chunk.data, GL_STATIC_DRAW)
    pb.chunks[i] = GPUChunk(vao, vbo, chunk.vertex_count, chunk.data)
  end
  pb.dirty = true
end

function create_vao()
  id = Ref{GLuint}(0)
  glGenVertexArrays(1, id)
  id[]
end

function create_vbo()
  id = Ref{GLuint}(0)
  glGenBuffers(1, id)
  id[]
end

function setup_interleaved_vao!(vao, vbo)
  glBindVertexArray(vao)
  glBindBuffer(GL_ARRAY_BUFFER, vbo)
  stride = Cint(VERTEX_FLOATS * sizeof(Float32))  # 40 bytes
  # Position at location 0, offset 0
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, Ptr{Cvoid}(0))
  glEnableVertexAttribArray(0)
  # Normal at location 1, offset 12 bytes
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, Ptr{Cvoid}(3 * sizeof(Float32)))
  glEnableVertexAttribArray(1)
  # Color at location 2, offset 24 bytes
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, Ptr{Cvoid}(6 * sizeof(Float32)))
  glEnableVertexAttribArray(2)
  glBindVertexArray(0)
end

# ─── Interactive camera callbacks ─────────────────────────────────────────────

function setup_callbacks(b::GL)
  GLFW.SetMouseButtonCallback(b.window, (_, button, action, _mods) -> begin
    if button == GLFW.MOUSE_BUTTON_LEFT
      b.orbit_active = action == GLFW.PRESS
    elseif button == GLFW.MOUSE_BUTTON_MIDDLE
      b.pan_active = action == GLFW.PRESS
    end
    if action == GLFW.PRESS
      x, y = GLFW.GetCursorPos(b.window)
      b.last_mouse_x = x
      b.last_mouse_y = y
    end
  end)

  GLFW.SetCursorPosCallback(b.window, (_, x, y) -> begin
    dx = x - b.last_mouse_x
    dy = y - b.last_mouse_y
    b.last_mouse_x = x
    b.last_mouse_y = y
    if b.orbit_active
      b.cam_azimuth -= dx * 0.005
      b.cam_elevation = clamp(b.cam_elevation + dy * 0.005, -π/2 + 0.01, π/2 - 0.01)
    elseif b.pan_active
      # Pan in the camera's local XY plane
      right = [-sin(b.cam_azimuth), cos(b.cam_azimuth), 0.0]
      up = [0.0, 0.0, 1.0]
      speed = b.cam_distance * 0.002
      b.cam_target .-= right .* (dx * speed)
      b.cam_target .+= up .* (dy * speed)
    end
  end)

  GLFW.SetScrollCallback(b.window, (_, _, dy) -> begin
    b.cam_distance = max(0.1, b.cam_distance * (1.0 - dy * 0.1))
  end)

  GLFW.SetKeyCallback(b.window, (_, key, _, action, _) -> begin
    if action == GLFW.PRESS
      if key == GLFW.KEY_ESCAPE
        GLFW.SetWindowShouldClose(b.window, true)
      elseif key == GLFW.KEY_R
        # Reset camera
        b.cam_distance = 30.0
        b.cam_azimuth = π/4
        b.cam_elevation = π/6
        b.cam_target .= 0.0
      elseif key == GLFW.KEY_W
        # Cycle display modes: shaded → shaded_wireframe → wireframe → shaded
        b.display_mode = if b.display_mode == :shaded
          :shaded_wireframe
        elseif b.display_mode == :shaded_wireframe
          :wireframe
        else
          :shaded
        end
      elseif key == GLFW.KEY_A
        # Toggle arctic mode
        b.display_mode = b.display_mode == :arctic ? :shaded : :arctic
      elseif key == GLFW.KEY_X
        # Toggle x-ray mode
        b.display_mode = b.display_mode == :xray ? :shaded : :xray
      elseif key == GLFW.KEY_M
        # Cycle MSAA: 0 → 2 → 4 → 8 → 0
        b.msaa_samples = b.msaa_samples == 0 ? 2 : b.msaa_samples == 2 ? 4 : b.msaa_samples == 4 ? 8 : 0
        b.needs_restart = true
        @info "MSAA set to $(b.msaa_samples)x — recreating window"
      elseif key == GLFW.KEY_F
        # Fit/zoom extents
        b_zoom_extents(b)
      elseif key == GLFW.KEY_1
        # Front view
        b.cam_azimuth = 0.0
        b.cam_elevation = 0.0
      elseif key == GLFW.KEY_2
        # Top view
        b.cam_azimuth = 0.0
        b.cam_elevation = π/2 - 0.01
      elseif key == GLFW.KEY_3
        # Perspective view (3/4)
        b.cam_azimuth = π/4
        b.cam_elevation = π/6
      end
    end
  end)
end

# ─── Camera and projection matrices ──────────────────────────────────────────

function camera_position(b::GL)
  let d = b.cam_distance,
      az = b.cam_azimuth,
      el = b.cam_elevation,
      t = b.cam_target
    [t[1] + d * cos(el) * cos(az),
     t[2] + d * cos(el) * sin(az),
     t[3] + d * sin(el)]
  end
end

function look_at_matrix(eye, target, up)
  f = normalize(target - eye)
  s = normalize(cross(f, up))
  u = cross(s, f)
  # Julia literals are row-major in appearance, stored column-major — matches OpenGL
  Float32[
    s[1]  s[2]  s[3]  -dot(s,eye);
    u[1]  u[2]  u[3]  -dot(u,eye);
   -f[1] -f[2] -f[3]   dot(f,eye);
    0     0     0      1
  ]
end

function perspective_matrix(fov_deg, aspect, near, far)
  fov = deg2rad(fov_deg)
  t = tan(fov / 2)
  Float32[
    1/(aspect*t) 0     0                          0;
    0            1/t   0                          0;
    0            0     -(far+near)/(far-near)    -2*far*near/(far-near);
    0            0     -1                         0
  ]
end

function model_matrix()
  Float32[
    1 0 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1
  ]
end

# ─── Coordinate and color conversion ─────────────────────────────────────────

gl_xyz(p) =
  let p = in_world(p)
    (Float32(p.x), Float32(p.y), Float32(p.z))
  end

const GL_DEFAULT_COLOR = (0.6f0, 0.6f0, 0.6f0, 1.0f0)

gl_color(::Nothing) = GL_DEFAULT_COLOR
gl_color(c::RGBA) = (Float32(c.r), Float32(c.g), Float32(c.b), Float32(c.alpha))
gl_color(c::RGB) = (Float32(c.r), Float32(c.g), Float32(c.b), 1.0f0)
gl_color(mat) =
  if hasfield(typeof(mat), :base_color) && mat.base_color !== nothing
    gl_color(mat.base_color)
  else
    GL_DEFAULT_COLOR
  end

# ─── Buffer append helpers ────────────────────────────────────────────────────

function next_ref!(b::GL)
  id = b.next_id
  b.next_id += 1
  GLNativeRef(id)
end

function append_trig_vertices!(scene::GLScene, x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                nx, ny, nz, r, g, b, a)
  pb = scene.tris
  if pb.offset + 30 > length(pb.buffer)  # 30 = 3 verts × 10 floats
    flush_buffer!(pb)
  end
  buf = pb.buffer
  o = pb.offset
  @inbounds begin
    buf[o+1]=x1;  buf[o+2]=y1;  buf[o+3]=z1
    buf[o+4]=nx;  buf[o+5]=ny;  buf[o+6]=nz
    buf[o+7]=r;   buf[o+8]=g;   buf[o+9]=b;   buf[o+10]=a
    buf[o+11]=x2; buf[o+12]=y2; buf[o+13]=z2
    buf[o+14]=nx; buf[o+15]=ny; buf[o+16]=nz
    buf[o+17]=r;  buf[o+18]=g;  buf[o+19]=b;  buf[o+20]=a
    buf[o+21]=x3; buf[o+22]=y3; buf[o+23]=z3
    buf[o+24]=nx; buf[o+25]=ny; buf[o+26]=nz
    buf[o+27]=r;  buf[o+28]=g;  buf[o+29]=b;  buf[o+30]=a
  end
  pb.offset = o + 30
  pb.dirty = true
end

function append_trig_vertices_smooth!(scene::GLScene,
    x1, y1, z1, x2, y2, z2, x3, y3, z3,
    n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z,
    r, g, b, a)
  pb = scene.tris
  if pb.offset + 30 > length(pb.buffer)
    flush_buffer!(pb)
  end
  buf = pb.buffer
  o = pb.offset
  @inbounds begin
    buf[o+1]=x1;  buf[o+2]=y1;  buf[o+3]=z1
    buf[o+4]=n1x; buf[o+5]=n1y; buf[o+6]=n1z
    buf[o+7]=r;   buf[o+8]=g;   buf[o+9]=b;   buf[o+10]=a
    buf[o+11]=x2; buf[o+12]=y2; buf[o+13]=z2
    buf[o+14]=n2x;buf[o+15]=n2y;buf[o+16]=n2z
    buf[o+17]=r;  buf[o+18]=g;  buf[o+19]=b;  buf[o+20]=a
    buf[o+21]=x3; buf[o+22]=y3; buf[o+23]=z3
    buf[o+24]=n3x;buf[o+25]=n3y;buf[o+26]=n3z
    buf[o+27]=r;  buf[o+28]=g;  buf[o+29]=b;  buf[o+30]=a
  end
  pb.offset = o + 30
  pb.dirty = true
end

function avg_normal(n1::NTuple{3,Float32}, n2::NTuple{3,Float32})
  let ax = n1[1] + n2[1],
      ay = n1[2] + n2[2],
      az = n1[3] + n2[3],
      len = sqrt(ax*ax + ay*ay + az*az)
    if len < 1.0f-10
      n1
    else
      (Float32(ax/len), Float32(ay/len), Float32(az/len))
    end
  end
end

function avg_normals(ns::Vector{NTuple{3,Float32}})
  let ax = sum(n -> n[1], ns),
      ay = sum(n -> n[2], ns),
      az = sum(n -> n[3], ns),
      len = sqrt(ax*ax + ay*ay + az*az)
    if len < 1.0f-10
      ns[1]
    else
      (Float32(ax/len), Float32(ay/len), Float32(az/len))
    end
  end
end

sub_tuple(a::NTuple{3,Float32}, b::NTuple{3,Float32}) =
  (a[1]-b[1], a[2]-b[2], a[3]-b[3])

normalize_tuple(v::NTuple{3,Float32}) =
  let len = sqrt(v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
    len < 1.0f-10 ? (0.0f0, 0.0f0, 1.0f0) : (v[1]/len, v[2]/len, v[3]/len)
  end

function compute_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3)
  # (p2-p1) × (p3-p1)
  ex, ey, ez = x2 - x1, y2 - y1, z2 - z1
  fx, fy, fz = x3 - x1, y3 - y1, z3 - z1
  nx = ey * fz - ez * fy
  ny = ez * fx - ex * fz
  nz = ex * fy - ey * fx
  len = sqrt(nx*nx + ny*ny + nz*nz)
  if len < 1.0f-10
    return (0.0f0, 0.0f0, 1.0f0)
  end
  (Float32(nx/len), Float32(ny/len), Float32(nz/len))
end

# ─── Geometry primitives ──────────────────────────────────────────────────────

KhepriBase.b_trig(b::GL, p1, p2, p3, mat) =
  let (x1, y1, z1) = gl_xyz(p1),
      (x2, y2, z2) = gl_xyz(p2),
      (x3, y3, z3) = gl_xyz(p3),
      (nx, ny, nz) = compute_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3),
      (r, g, bl, a) = gl_color(mat)
    append_trig_vertices!(b.scene, x1, y1, z1, x2, y2, z2, x3, y3, z3,
                          nx, ny, nz, r, g, bl, a)
    next_ref!(b)
  end

KhepriBase.b_quad(b::GL, p1, p2, p3, p4, mat) =
  let (x1, y1, z1) = gl_xyz(p1),
      (x2, y2, z2) = gl_xyz(p2),
      (x3, y3, z3) = gl_xyz(p3),
      (x4, y4, z4) = gl_xyz(p4),
      (r, g, bl, a) = gl_color(mat),
      (n1x, n1y, n1z) = compute_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3),
      (n2x, n2y, n2z) = compute_normal(x1, y1, z1, x3, y3, z3, x4, y4, z4)
    append_trig_vertices!(b.scene, x1, y1, z1, x2, y2, z2, x3, y3, z3,
                          n1x, n1y, n1z, r, g, bl, a)
    append_trig_vertices!(b.scene, x1, y1, z1, x3, y3, z3, x4, y4, z4,
                          n2x, n2y, n2z, r, g, bl, a)
    next_ref!(b)
  end

KhepriBase.b_ngon(b::GL, ps, pivot, smooth, mat) =
  let (px, py, pz) = gl_xyz(pivot),
      n = length(ps),
      coords = [gl_xyz(p) for p in ps],
      (r, g, bl, a) = gl_color(mat)
    if smooth
      let face_normals = [compute_normal(px, py, pz, coords[i]..., coords[i % n + 1]...) for i in 1:n],
          pivot_normal = avg_normals(face_normals)
        for i in 1:n
          let j = i % n + 1,
              (x1, y1, z1) = coords[i],
              (x2, y2, z2) = coords[j],
              prev = mod1(i - 1, n),
              vn1 = avg_normal(face_normals[prev], face_normals[i]),
              vn2 = avg_normal(face_normals[i], face_normals[j])
            append_trig_vertices_smooth!(b.scene, px, py, pz, x1, y1, z1, x2, y2, z2,
                                         pivot_normal..., vn1..., vn2...,
                                         r, g, bl, a)
          end
        end
      end
    else
      for i in 1:n
        let (x1, y1, z1) = coords[i],
            (x2, y2, z2) = coords[i % n + 1],
            (nx, ny, nz) = compute_normal(px, py, pz, x1, y1, z1, x2, y2, z2)
          append_trig_vertices!(b.scene, px, py, pz, x1, y1, z1, x2, y2, z2,
                                nx, ny, nz, r, g, bl, a)
        end
      end
    end
    next_ref!(b)
  end

KhepriBase.b_quad_strip(b::GL, ps, qs, smooth, mat) =
  let n = length(ps),
      pcoords = [gl_xyz(p) for p in ps],
      qcoords = [gl_xyz(q) for q in qs],
      (r, g, bl, a) = gl_color(mat)
    if smooth
      # Precompute face normals for each quad (average of its two triangle normals)
      let face_normals = [
            let (px1, py1, pz1) = pcoords[i],
                (px2, py2, pz2) = pcoords[i+1],
                (qx2, qy2, qz2) = qcoords[i+1],
                (qx1, qy1, qz1) = qcoords[i],
                fn1 = compute_normal(px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2),
                fn2 = compute_normal(px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1)
              avg_normal(fn1, fn2)
            end
            for i in 1:n-1]
        for i in 1:n-1
          let (px1, py1, pz1) = pcoords[i],
              (px2, py2, pz2) = pcoords[i+1],
              (qx2, qy2, qz2) = qcoords[i+1],
              (qx1, qy1, qz1) = qcoords[i],
              # Vertex normals: average of adjacent face normals (endpoints use single face normal)
              nl = i == 1   ? face_normals[1]   : avg_normal(face_normals[i-1], face_normals[i]),
              nr = i == n-1 ? face_normals[n-1] : avg_normal(face_normals[i], face_normals[i+1])
            # Triangle 1: p[i], p[i+1], q[i+1] — normals: nl, nr, nr
            append_trig_vertices_smooth!(b.scene,
              px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2,
              nl..., nr..., nr...,
              r, g, bl, a)
            # Triangle 2: p[i], q[i+1], q[i] — normals: nl, nr, nl
            append_trig_vertices_smooth!(b.scene,
              px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1,
              nl..., nr..., nl...,
              r, g, bl, a)
          end
        end
      end
    else
      for i in 1:n-1
        let (px1, py1, pz1) = pcoords[i],
            (px2, py2, pz2) = pcoords[i+1],
            (qx2, qy2, qz2) = qcoords[i+1],
            (qx1, qy1, qz1) = qcoords[i],
            (n1x, n1y, n1z) = compute_normal(px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2),
            (n2x, n2y, n2z) = compute_normal(px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1)
          append_trig_vertices!(b.scene, px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2,
                                n1x, n1y, n1z, r, g, bl, a)
          append_trig_vertices!(b.scene, px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1,
                                n2x, n2y, n2z, r, g, bl, a)
        end
      end
    end
    next_ref!(b)
  end

KhepriBase.b_quad_strip_closed(b::GL, ps, qs, smooth, mat) =
  let n = length(ps),
      pcoords = [gl_xyz(p) for p in ps],
      qcoords = [gl_xyz(q) for q in qs],
      (r, g, bl, a) = gl_color(mat)
    if smooth
      # Precompute face normals for each quad (average of its two triangle normals)
      let face_normals = [
            let j = i % n + 1,
                (px1, py1, pz1) = pcoords[i],
                (px2, py2, pz2) = pcoords[j],
                (qx2, qy2, qz2) = qcoords[j],
                (qx1, qy1, qz1) = qcoords[i],
                fn1 = compute_normal(px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2),
                fn2 = compute_normal(px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1)
              avg_normal(fn1, fn2)
            end
            for i in 1:n]
        for i in 1:n
          let j = i % n + 1,
              (px1, py1, pz1) = pcoords[i],
              (px2, py2, pz2) = pcoords[j],
              (qx2, qy2, qz2) = qcoords[j],
              (qx1, qy1, qz1) = qcoords[i],
              prev = mod1(i - 1, n),
              # Vertex normals: average of this face and the adjacent face (wrapping)
              nl = avg_normal(face_normals[prev], face_normals[i]),
              nr = avg_normal(face_normals[i], face_normals[j])
            # Triangle 1: p[i], p[j], q[j] — normals: nl, nr, nr
            append_trig_vertices_smooth!(b.scene,
              px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2,
              nl..., nr..., nr...,
              r, g, bl, a)
            # Triangle 2: p[i], q[j], q[i] — normals: nl, nr, nl
            append_trig_vertices_smooth!(b.scene,
              px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1,
              nl..., nr..., nl...,
              r, g, bl, a)
          end
        end
      end
    else
      for i in 1:n
        let j = i % n + 1,
            (px1, py1, pz1) = pcoords[i],
            (px2, py2, pz2) = pcoords[j],
            (qx2, qy2, qz2) = qcoords[j],
            (qx1, qy1, qz1) = qcoords[i],
            (n1x, n1y, n1z) = compute_normal(px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2),
            (n2x, n2y, n2z) = compute_normal(px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1)
          append_trig_vertices!(b.scene, px1, py1, pz1, px2, py2, pz2, qx2, qy2, qz2,
                                n1x, n1y, n1z, r, g, bl, a)
          append_trig_vertices!(b.scene, px1, py1, pz1, qx2, qy2, qz2, qx1, qy1, qz1,
                                n2x, n2y, n2z, r, g, bl, a)
        end
      end
    end
    next_ref!(b)
  end

function append_line_vertices!(scene::GLScene, x1, y1, z1, x2, y2, z2, r, g, b, a)
  pb = scene.lines
  if pb.offset + 20 > length(pb.buffer)  # 20 = 2 verts × 10 floats
    flush_buffer!(pb)
  end
  buf = pb.buffer
  o = pb.offset
  @inbounds begin
    buf[o+1]=x1; buf[o+2]=y1; buf[o+3]=z1
    buf[o+4]=0f0; buf[o+5]=0f0; buf[o+6]=0f0  # dummy normal
    buf[o+7]=r; buf[o+8]=g; buf[o+9]=b; buf[o+10]=a
    buf[o+11]=x2; buf[o+12]=y2; buf[o+13]=z2
    buf[o+14]=0f0; buf[o+15]=0f0; buf[o+16]=0f0
    buf[o+17]=r; buf[o+18]=g; buf[o+19]=b; buf[o+20]=a
  end
  pb.offset = o + 20
  pb.dirty = true
end

function append_point_vertex!(scene::GLScene, x, y, z, r, g, b, a)
  pb = scene.points
  if pb.offset + 10 > length(pb.buffer)
    flush_buffer!(pb)
  end
  buf = pb.buffer
  o = pb.offset
  @inbounds begin
    buf[o+1]=x; buf[o+2]=y; buf[o+3]=z
    buf[o+4]=0f0; buf[o+5]=0f0; buf[o+6]=0f0
    buf[o+7]=r; buf[o+8]=g; buf[o+9]=b; buf[o+10]=a
  end
  pb.offset = o + 10
  pb.dirty = true
end

KhepriBase.b_line(b::GL, ps, mat) =
  let (r, g, bl, a) = gl_color(mat),
      n = length(ps)
    for i in 1:n-1
      let (x1, y1, z1) = gl_xyz(ps[i]),
          (x2, y2, z2) = gl_xyz(ps[i+1])
        append_line_vertices!(b.scene, x1, y1, z1, x2, y2, z2, r, g, bl, a)
      end
    end
    next_ref!(b)
  end

KhepriBase.b_point(b::GL, p, mat) =
  let (x, y, z) = gl_xyz(p),
      (r, g, bl, a) = gl_color(mat)
    append_point_vertex!(b.scene, x, y, z, r, g, bl, a)
    next_ref!(b)
  end

KhepriBase.b_surface_polygon(b::GL, ps, mat) =
  let n = length(ps)
    if n < 3
      void_ref(b)
    elseif n == 3
      b_trig(b, ps[1], ps[2], ps[3], mat)
    else
      # Fan triangulation from first vertex
      let (r, g, bl, a) = gl_color(mat),
          coords = [gl_xyz(p) for p in ps]
        for i in 2:n-1
          let (x1, y1, z1) = coords[1],
              (x2, y2, z2) = coords[i],
              (x3, y3, z3) = coords[i+1],
              (nx, ny, nz) = compute_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3)
            append_trig_vertices!(b.scene, x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                  nx, ny, nz, r, g, bl, a)
          end
        end
        next_ref!(b)
      end
    end
  end

KhepriBase.b_surface_polygon_with_holes(b::GL, ps, qss, mat) =
  # Simplified: render outer polygon (proper hole triangulation would need earcut)
  b_surface_polygon(b, ps, mat)

KhepriBase.b_surface_circle(b::GL, c, r, mat) =
  b_surface_polygon(b, [c + vpol(r, θ, c.cs) for θ in range(0, 2π, length=33)[1:32]], mat)

KhepriBase.b_surface_arc(b::GL, c, r, α, Δα, mat) =
  let pts = [c + vpol(r, θ, c.cs) for θ in range(α, α + Δα, length=max(2, abs(round(Int, 32 * Δα / (2π))) + 1))]
    b_surface_polygon(b, [c, pts...], mat)
  end

# ─── Direct sphere with analytic normals ─────────────────────────────────────

KhepriBase.b_sphere(b::GL, c, r, mat) =
  let nlon = 32,
      nlat = 16,
      (cr, cg, cb, ca) = gl_color(mat),
      center = gl_xyz(c),
      # Precompute latitude rings: ring[j] has nlon vertices at polar angle ψ = j*π/nlat
      rings = [
        let ψ = j * π / nlat,
            sinψ = Float32(sin(ψ)),
            cosψ = Float32(cos(ψ))
          [let ϕ = (i-1) * 2π / nlon,
               x = center[1] + Float32(r) * Float32(cos(ϕ)) * sinψ,
               y = center[2] + Float32(r) * Float32(sin(ϕ)) * sinψ,
               z = center[3] + Float32(r) * cosψ
             (x, y, z)
           end
           for i in 1:nlon]
        end
        for j in 1:nlat-1],
      north = (center[1], center[2], center[3] + Float32(r)),
      south = (center[1], center[2], center[3] - Float32(r)),
      north_n = (0.0f0, 0.0f0, 1.0f0),
      south_n = (0.0f0, 0.0f0, -1.0f0),
      scene = b.scene
    # Analytic normal for a vertex on the sphere
    analytic_n(v) = normalize_tuple(sub_tuple(v, center))
    # Top cap: triangle fan from north pole to first ring
    for i in 1:nlon
      let j = i % nlon + 1,
          v1 = rings[1][i],
          v2 = rings[1][j]
        append_trig_vertices_smooth!(scene,
          north..., v1..., v2...,
          north_n..., analytic_n(v1)..., analytic_n(v2)...,
          cr, cg, cb, ca)
      end
    end
    # Middle bands: quads between adjacent rings
    for k in 1:nlat-2
      for i in 1:nlon
        let j = i % nlon + 1,
            p1 = rings[k][i],
            p2 = rings[k][j],
            q1 = rings[k+1][i],
            q2 = rings[k+1][j]
          # Triangle 1: p1, p2, q2
          append_trig_vertices_smooth!(scene,
            p1..., p2..., q2...,
            analytic_n(p1)..., analytic_n(p2)..., analytic_n(q2)...,
            cr, cg, cb, ca)
          # Triangle 2: p1, q2, q1
          append_trig_vertices_smooth!(scene,
            p1..., q2..., q1...,
            analytic_n(p1)..., analytic_n(q2)..., analytic_n(q1)...,
            cr, cg, cb, ca)
        end
      end
    end
    # Bottom cap: triangle fan from last ring to south pole
    for i in 1:nlon
      let j = i % nlon + 1,
          v1 = rings[nlat-1][i],
          v2 = rings[nlat-1][j]
        append_trig_vertices_smooth!(scene,
          v1..., v2..., south...,
          analytic_n(v1)..., analytic_n(v2)..., south_n...,
          cr, cg, cb, ca)
      end
    end
    next_ref!(b)
  end

# ─── Direct cone with analytic normals ───────────────────────────────────────

KhepriBase.b_cone(b::GL, cb, r, h, mat) =
  let n = 32,
      (cr, cg, cbl, ca) = gl_color(mat),
      base_c = gl_xyz(cb),
      apex = gl_xyz(add_z(cb, h)),
      base_vs = [gl_xyz(cb + vpol(r, (i-1) * 2π / n, cb.cs)) for i in 1:n],
      # Axis direction and height
      axis_raw = sub_tuple(apex, base_c),
      ht = sqrt(axis_raw[1]^2 + axis_raw[2]^2 + axis_raw[3]^2),
      axis = ht < 1.0f-10 ? (0.0f0, 0.0f0, 1.0f0) : (axis_raw[1]/ht, axis_raw[2]/ht, axis_raw[3]/ht),
      dr = Float32(r),
      # Lateral normals: normalize(ht * r̂ + r * axis) (cone_frustum formula with rt=0)
      lat_normals = [
        let r_dir = sub_tuple(base_vs[i], base_c),
            r_len = sqrt(r_dir[1]^2 + r_dir[2]^2 + r_dir[3]^2),
            rhat = r_len < 1.0f-10 ? (0.0f0, 0.0f0, 0.0f0) : (r_dir[1]/r_len, r_dir[2]/r_len, r_dir[3]/r_len)
          normalize_tuple((ht * rhat[1] + dr * axis[1],
                           ht * rhat[2] + dr * axis[2],
                           ht * rhat[3] + dr * axis[3]))
        end
        for i in 1:n],
      base_n = (-axis[1], -axis[2], -axis[3]),
      scene = b.scene
    # Lateral surface: triangle fan from base ring to apex
    for i in 1:n
      let j = i % n + 1,
          apex_n = avg_normal(lat_normals[i], lat_normals[j])
        append_trig_vertices_smooth!(scene,
          base_vs[i]..., base_vs[j]..., apex...,
          lat_normals[i]..., lat_normals[j]..., apex_n...,
          cr, cg, cbl, ca)
      end
    end
    # Base cap
    for i in 1:n
      let j = i % n + 1
        append_trig_vertices!(scene,
          base_c..., base_vs[j]..., base_vs[i]...,
          base_n..., cr, cg, cbl, ca)
      end
    end
    next_ref!(b)
  end

# ─── Direct cuboid and box with flat normals ──────────────────────────────────

function emit_cuboid!(scene, p0, p1, p2, p3, p4, p5, p6, p7, cr, cg, cbl, ca)
  # 6 faces, 2 triangles each, flat normals via compute_normal
  # Bottom face (p0, p3, p2, p1)
  let (nx, ny, nz) = compute_normal(p0..., p3..., p2...)
    append_trig_vertices!(scene, p0..., p3..., p2..., nx, ny, nz, cr, cg, cbl, ca)
    append_trig_vertices!(scene, p0..., p2..., p1..., nx, ny, nz, cr, cg, cbl, ca)
  end
  # Top face (p4, p5, p6, p7)
  let (nx, ny, nz) = compute_normal(p4..., p5..., p6...)
    append_trig_vertices!(scene, p4..., p5..., p6..., nx, ny, nz, cr, cg, cbl, ca)
    append_trig_vertices!(scene, p4..., p6..., p7..., nx, ny, nz, cr, cg, cbl, ca)
  end
  # Front face (p0, p1, p5, p4)
  let (nx, ny, nz) = compute_normal(p0..., p1..., p5...)
    append_trig_vertices!(scene, p0..., p1..., p5..., nx, ny, nz, cr, cg, cbl, ca)
    append_trig_vertices!(scene, p0..., p5..., p4..., nx, ny, nz, cr, cg, cbl, ca)
  end
  # Back face (p3, p7, p6, p2)
  let (nx, ny, nz) = compute_normal(p3..., p7..., p6...)
    append_trig_vertices!(scene, p3..., p7..., p6..., nx, ny, nz, cr, cg, cbl, ca)
    append_trig_vertices!(scene, p3..., p6..., p2..., nx, ny, nz, cr, cg, cbl, ca)
  end
  # Left face (p0, p4, p7, p3)
  let (nx, ny, nz) = compute_normal(p0..., p4..., p7...)
    append_trig_vertices!(scene, p0..., p4..., p7..., nx, ny, nz, cr, cg, cbl, ca)
    append_trig_vertices!(scene, p0..., p7..., p3..., nx, ny, nz, cr, cg, cbl, ca)
  end
  # Right face (p1, p2, p6, p5)
  let (nx, ny, nz) = compute_normal(p1..., p2..., p6...)
    append_trig_vertices!(scene, p1..., p2..., p6..., nx, ny, nz, cr, cg, cbl, ca)
    append_trig_vertices!(scene, p1..., p6..., p5..., nx, ny, nz, cr, cg, cbl, ca)
  end
end

KhepriBase.b_cuboid(b::GL, pb0, pb1, pb2, pb3, pt0, pt1, pt2, pt3, mat) =
  let (cr, cg, cbl, ca) = gl_color(mat)
    emit_cuboid!(b.scene,
      gl_xyz(pb0), gl_xyz(pb1), gl_xyz(pb2), gl_xyz(pb3),
      gl_xyz(pt0), gl_xyz(pt1), gl_xyz(pt2), gl_xyz(pt3),
      cr, cg, cbl, ca)
    next_ref!(b)
  end

KhepriBase.b_box(b::GL, c, dx, dy, dz, mat) =
  let (cr, cg, cbl, ca) = gl_color(mat)
    emit_cuboid!(b.scene,
      gl_xyz(c),                       gl_xyz(add_x(c, dx)),
      gl_xyz(add_xy(c, dx, dy)),       gl_xyz(add_y(c, dy)),
      gl_xyz(add_z(c, dz)),            gl_xyz(add_xyz(c, dx, 0, dz)),
      gl_xyz(add_xyz(c, dx, dy, dz)),  gl_xyz(add_xyz(c, 0, dy, dz)),
      cr, cg, cbl, ca)
    next_ref!(b)
  end

# ─── Direct regular_pyramid with flat normals ────────────────────────────────

KhepriBase.b_regular_pyramid(b::GL, edges, cb, rb, angle, h, inscribed, mat) =
  let n = edges,
      (cr, cg, cbl, ca) = gl_color(mat),
      base_c = gl_xyz(cb),
      apex = gl_xyz(add_z(cb, h)),
      base_vs = [gl_xyz(p) for p in regular_polygon_vertices(n, cb, rb, angle, inscribed)],
      # Axis for base cap normal
      axis_raw = sub_tuple(apex, base_c),
      ht = sqrt(axis_raw[1]^2 + axis_raw[2]^2 + axis_raw[3]^2),
      axis = ht < 1.0f-10 ? (0.0f0, 0.0f0, 1.0f0) : (axis_raw[1]/ht, axis_raw[2]/ht, axis_raw[3]/ht),
      base_n = (-axis[1], -axis[2], -axis[3]),
      scene = b.scene
    # Lateral faces (flat shading — each triangular face has its own normal)
    for i in 1:n
      let j = i % n + 1,
          (nx, ny, nz) = compute_normal(base_vs[i]..., base_vs[j]..., apex...)
        append_trig_vertices!(scene,
          base_vs[i]..., base_vs[j]..., apex...,
          nx, ny, nz, cr, cg, cbl, ca)
      end
    end
    # Base cap
    for i in 1:n
      let j = i % n + 1
        append_trig_vertices!(scene,
          base_c..., base_vs[j]..., base_vs[i]...,
          base_n..., cr, cg, cbl, ca)
      end
    end
    next_ref!(b)
  end

# ─── Direct regular_pyramid_frustum with flat normals ────────────────────────

KhepriBase.b_regular_pyramid_frustum(b::GL, edges, cb, rb, angle, h, rt, inscribed, mat) =
  let n = edges,
      (cr, cg, cbl, ca) = gl_color(mat),
      base_c = gl_xyz(cb),
      top_c = gl_xyz(add_z(cb, h)),
      base_vs = [gl_xyz(p) for p in regular_polygon_vertices(n, cb, rb, angle, inscribed)],
      top_vs = [gl_xyz(p) for p in regular_polygon_vertices(n, add_z(cb, h), rt, angle, inscribed)],
      # Axis for cap normals
      axis_raw = sub_tuple(top_c, base_c),
      ht = sqrt(axis_raw[1]^2 + axis_raw[2]^2 + axis_raw[3]^2),
      axis = ht < 1.0f-10 ? (0.0f0, 0.0f0, 1.0f0) : (axis_raw[1]/ht, axis_raw[2]/ht, axis_raw[3]/ht),
      base_n = (-axis[1], -axis[2], -axis[3]),
      top_n = axis,
      scene = b.scene
    # Lateral faces (flat shading — each quad face gets its own normal)
    for i in 1:n
      let j = i % n + 1,
          bv1 = base_vs[i], bv2 = base_vs[j],
          tv1 = top_vs[i], tv2 = top_vs[j],
          (nx, ny, nz) = compute_normal(bv1..., bv2..., tv2...)
        append_trig_vertices!(scene,
          bv1..., bv2..., tv2...,
          nx, ny, nz, cr, cg, cbl, ca)
        append_trig_vertices!(scene,
          bv1..., tv2..., tv1...,
          nx, ny, nz, cr, cg, cbl, ca)
      end
    end
    # Base cap
    for i in 1:n
      let j = i % n + 1
        append_trig_vertices!(scene,
          base_c..., base_vs[j]..., base_vs[i]...,
          base_n..., cr, cg, cbl, ca)
      end
    end
    # Top cap
    for i in 1:n
      let j = i % n + 1
        append_trig_vertices!(scene,
          top_c..., top_vs[i]..., top_vs[j]...,
          top_n..., cr, cg, cbl, ca)
      end
    end
    next_ref!(b)
  end

KhepriBase.b_regular_prism(b::GL, edges, cb, rb, angle, h, inscribed, mat) =
  b_regular_pyramid_frustum(b, edges, cb, rb, angle, h, rb, inscribed, mat)

# ─── Direct cone_frustum and cylinder with analytic normals ──────────────────

KhepriBase.b_cone_frustum(b::GL, cb, rb, h, rt, mat) =
  let n = 32,
      (cr, cg, cbl, ca) = gl_color(mat),
      base_c = gl_xyz(cb),
      top_c = gl_xyz(add_z(cb, h)),
      base_vs = [gl_xyz(cb + vpol(rb, (i-1) * 2π / n, cb.cs)) for i in 1:n],
      top_vs = [gl_xyz(add_z(cb, h) + vpol(rt, (i-1) * 2π / n, cb.cs)) for i in 1:n],
      # Axis direction and height
      axis_raw = sub_tuple(top_c, base_c),
      ht = sqrt(axis_raw[1]^2 + axis_raw[2]^2 + axis_raw[3]^2),
      axis = ht < 1.0f-10 ? (0.0f0, 0.0f0, 1.0f0) : (axis_raw[1]/ht, axis_raw[2]/ht, axis_raw[3]/ht),
      dr = Float32(rb - rt),
      # Precompute lateral normals (one per angular position)
      # n = normalize(ht * r̂ + (rb - rt) * â) where r̂ is unit radial
      lat_normals = [
        let r_dir = sub_tuple(base_vs[i], base_c),
            r_len = sqrt(r_dir[1]^2 + r_dir[2]^2 + r_dir[3]^2),
            rhat = r_len < 1.0f-10 ? (0.0f0, 0.0f0, 0.0f0) : (r_dir[1]/r_len, r_dir[2]/r_len, r_dir[3]/r_len)
          normalize_tuple((ht * rhat[1] + dr * axis[1],
                           ht * rhat[2] + dr * axis[2],
                           ht * rhat[3] + dr * axis[3]))
        end
        for i in 1:n],
      base_n = (-axis[1], -axis[2], -axis[3]),
      top_n = axis,
      scene = b.scene
    # Lateral surface
    for i in 1:n
      let j = i % n + 1
        append_trig_vertices_smooth!(scene,
          base_vs[i]..., base_vs[j]..., top_vs[j]...,
          lat_normals[i]..., lat_normals[j]..., lat_normals[j]...,
          cr, cg, cbl, ca)
        append_trig_vertices_smooth!(scene,
          base_vs[i]..., top_vs[j]..., top_vs[i]...,
          lat_normals[i]..., lat_normals[j]..., lat_normals[i]...,
          cr, cg, cbl, ca)
      end
    end
    # Base cap
    if rb > 0
      for i in 1:n
        let j = i % n + 1
          append_trig_vertices!(scene,
            base_c..., base_vs[j]..., base_vs[i]...,
            base_n..., cr, cg, cbl, ca)
        end
      end
    end
    # Top cap
    if rt > 0
      for i in 1:n
        let j = i % n + 1
          append_trig_vertices!(scene,
            top_c..., top_vs[i]..., top_vs[j]...,
            top_n..., cr, cg, cbl, ca)
        end
      end
    end
    next_ref!(b)
  end

KhepriBase.b_cylinder(b::GL, cb, r, h, mat) =
  KhepriBase.b_cone_frustum(b, cb, r, h, r, mat)

# ─── Direct torus with analytic normals ──────────────────────────────────────

KhepriBase.b_torus(b::GL, c, ra, rb, mat) =
  let ntor = 64,  # toroidal divisions (around the major circle)
      npol = 32,  # poloidal divisions (around the tube cross-section)
      (cr, cg, cb, ca) = gl_color(mat),
      center = gl_xyz(c),
      scene = b.scene,
      # Precompute all vertices and normals
      # vertex[i,j] = point on torus at toroidal angle ϕ_i, poloidal angle ψ_j
      # Normal = direction from tube center to vertex (analytic)
      vertices = Matrix{NTuple{3,Float32}}(undef, ntor, npol),
      normals = Matrix{NTuple{3,Float32}}(undef, ntor, npol)
    for i in 1:ntor
      let ϕ = (i-1) * 2π / ntor,
          cosϕ = Float32(cos(ϕ)),
          sinϕ = Float32(sin(ϕ)),
          # Center of tube cross-section at this toroidal angle
          tcx = center[1] + Float32(ra) * cosϕ,
          tcy = center[2] + Float32(ra) * sinϕ,
          tcz = center[3]
        for j in 1:npol
          let ψ = (j-1) * 2π / npol,
              cosψ = Float32(cos(ψ)),
              sinψ = Float32(sin(ψ)),
              # Point on the tube surface
              x = tcx + Float32(rb) * cosψ * cosϕ,
              y = tcy + Float32(rb) * cosψ * sinϕ,
              z = tcz + Float32(rb) * sinψ,
              # Normal: direction from tube center to point (analytic)
              nx = cosψ * cosϕ,
              ny = cosψ * sinϕ,
              nz = sinψ
            vertices[i,j] = (x, y, z)
            normals[i,j] = (Float32(nx), Float32(ny), Float32(nz))
          end
        end
      end
    end
    # Emit quads for all toroidal × poloidal cells (both directions closed)
    for i in 1:ntor
      let i2 = i % ntor + 1
        for j in 1:npol
          let j2 = j % npol + 1,
              p00 = vertices[i, j],
              p10 = vertices[i2, j],
              p11 = vertices[i2, j2],
              p01 = vertices[i, j2],
              n00 = normals[i, j],
              n10 = normals[i2, j],
              n11 = normals[i2, j2],
              n01 = normals[i, j2]
            append_trig_vertices_smooth!(scene,
              p00..., p10..., p11...,
              n00..., n10..., n11...,
              cr, cg, cb, ca)
            append_trig_vertices_smooth!(scene,
              p00..., p11..., p01...,
              n00..., n11..., n01...,
              cr, cg, cb, ca)
          end
        end
      end
    end
    next_ref!(b)
  end

# ─── Direct surface grid with cross-strip normals ────────────────────────────

function emit_grid_smooth!(scene, coords, nu, nv, closed_u, closed_v, r, g, bl, a)
  # coords is a nu×nv matrix of (Float32, Float32, Float32) tuples
  # Compute face normals for all quads
  # Face (i,j) is the quad with corners (i,j), (i+1,j), (i+1,j+1), (i,j+1)
  # Number of faces in each direction
  nfu = closed_u ? nu : nu - 1
  nfv = closed_v ? nv : nv - 1
  face_normals = Matrix{NTuple{3,Float32}}(undef, nfu, nfv)
  for fi in 1:nfu
    for fj in 1:nfv
      let i0 = fi,
          i1 = closed_u ? (fi % nu + 1) : fi + 1,
          j0 = fj,
          j1 = closed_v ? (fj % nv + 1) : fj + 1,
          p00 = coords[i0, j0],
          p10 = coords[i1, j0],
          p11 = coords[i1, j1],
          p01 = coords[i0, j1],
          fn1 = compute_normal(p00..., p10..., p11...),
          fn2 = compute_normal(p00..., p11..., p01...)
        face_normals[fi, fj] = avg_normal(fn1, fn2)
      end
    end
  end
  # Compute per-vertex normals by averaging all adjacent face normals (up to 4)
  # Accumulate sum directly to avoid per-vertex heap allocations
  vertex_normals = Matrix{NTuple{3,Float32}}(undef, nu, nv)
  for vi in 1:nu
    for vj in 1:nv
      let sx = 0.0f0, sy = 0.0f0, sz = 0.0f0, cnt = 0
        # Face (vi, vj) — vertex is at top-left corner
        if (closed_u || vi <= nfu) && (closed_v || vj <= nfv)
          let fn = face_normals[closed_u ? mod1(vi, nfu) : vi,
                                closed_v ? mod1(vj, nfv) : vj]
            sx += fn[1]; sy += fn[2]; sz += fn[3]; cnt += 1
          end
        end
        # Face (vi-1, vj) — vertex is at top-right corner
        if (closed_u || vi > 1) && (closed_v || vj <= nfv)
          let fn = face_normals[closed_u ? mod1(vi - 1, nfu) : vi - 1,
                                closed_v ? mod1(vj, nfv) : vj]
            sx += fn[1]; sy += fn[2]; sz += fn[3]; cnt += 1
          end
        end
        # Face (vi-1, vj-1) — vertex is at bottom-right corner
        if (closed_u || vi > 1) && (closed_v || vj > 1)
          let fn = face_normals[closed_u ? mod1(vi - 1, nfu) : vi - 1,
                                closed_v ? mod1(vj - 1, nfv) : vj - 1]
            sx += fn[1]; sy += fn[2]; sz += fn[3]; cnt += 1
          end
        end
        # Face (vi, vj-1) — vertex is at bottom-left corner
        if (closed_u || vi <= nfu) && (closed_v || vj > 1)
          let fn = face_normals[closed_u ? mod1(vi, nfu) : vi,
                                closed_v ? mod1(vj - 1, nfv) : vj - 1]
            sx += fn[1]; sy += fn[2]; sz += fn[3]; cnt += 1
          end
        end
        vertex_normals[vi, vj] = cnt == 0 ?
          (0.0f0, 0.0f0, 1.0f0) :
          normalize_tuple((sx, sy, sz))
      end
    end
  end
  # Emit triangles for all quads
  for fi in 1:nfu
    for fj in 1:nfv
      let i0 = fi,
          i1 = closed_u ? (fi % nu + 1) : fi + 1,
          j0 = fj,
          j1 = closed_v ? (fj % nv + 1) : fj + 1,
          p00 = coords[i0, j0],
          p10 = coords[i1, j0],
          p11 = coords[i1, j1],
          p01 = coords[i0, j1],
          n00 = vertex_normals[i0, j0],
          n10 = vertex_normals[i1, j0],
          n11 = vertex_normals[i1, j1],
          n01 = vertex_normals[i0, j1]
        # Triangle 1: p00, p10, p11
        append_trig_vertices_smooth!(scene,
          p00..., p10..., p11...,
          n00..., n10..., n11...,
          r, g, bl, a)
        # Triangle 2: p00, p11, p01
        append_trig_vertices_smooth!(scene,
          p00..., p11..., p01...,
          n00..., n11..., n01...,
          r, g, bl, a)
      end
    end
  end
end

function emit_grid_flat!(scene, coords, nu, nv, closed_u, closed_v, r, g, bl, a)
  nfu = closed_u ? nu : nu - 1
  nfv = closed_v ? nv : nv - 1
  for fi in 1:nfu
    for fj in 1:nfv
      let i0 = fi,
          i1 = closed_u ? (fi % nu + 1) : fi + 1,
          j0 = fj,
          j1 = closed_v ? (fj % nv + 1) : fj + 1,
          p00 = coords[i0, j0],
          p10 = coords[i1, j0],
          p11 = coords[i1, j1],
          p01 = coords[i0, j1],
          (n1x, n1y, n1z) = compute_normal(p00..., p10..., p11...),
          (n2x, n2y, n2z) = compute_normal(p00..., p11..., p01...)
        append_trig_vertices!(scene, p00..., p10..., p11...,
                              n1x, n1y, n1z, r, g, bl, a)
        append_trig_vertices!(scene, p00..., p11..., p01...,
                              n2x, n2y, n2z, r, g, bl, a)
      end
    end
  end
end

KhepriBase.b_surface_grid(b::GL, ptss, closed_u, closed_v, smooth_u, smooth_v, mat) =
  let ptss = maybe_interpolate_grid(ptss, smooth_u, smooth_v),
      (nu, nv) = size(ptss),
      coords = [gl_xyz(ptss[i, j]) for i in 1:nu, j in 1:nv],
      (r, g, bl, a) = gl_color(mat),
      smooth = smooth_u || smooth_v
    if smooth
      emit_grid_smooth!(b.scene, coords, nu, nv, closed_u, closed_v, r, g, bl, a)
    else
      emit_grid_flat!(b.scene, coords, nu, nv, closed_u, closed_v, r, g, bl, a)
    end
    next_ref!(b)
  end

# ─── Material handling ────────────────────────────────────────────────────────

KhepriBase.b_new_material(b::GL, name, base_color, metallic, specular, roughness,
                          clearcoat, clearcoat_roughness, ior, transmission,
                          transmission_roughness, emission_color, emission_strength) =
  base_color

KhepriBase.b_plastic_material(b::GL, name, color, roughness) = color
KhepriBase.b_metal_material(b::GL, name, color, roughness, ior) = color
KhepriBase.b_glass_material(b::GL, name, color, roughness, ior) = color
KhepriBase.b_mirror_material(b::GL, name, color) = color

# ─── Layer operations (minimal) ──────────────────────────────────────────────

KhepriBase.b_layer(b::GL, name, active, color) = name
KhepriBase.b_current_layer_ref(b::GL) = "default"
KhepriBase.b_current_layer_ref(b::GL, layer) = layer
KhepriBase.b_delete_all_shapes_in_layer(b::GL, layer) = nothing

# ─── Boolean operations (visual approximation) ──────────────────────────────
# True CSG is not supported; both shapes are simply rendered.

KhepriBase.b_subtract_ref(b::GL, sref, mref) = sref
KhepriBase.b_intersect_ref(b::GL, sref, mref) = sref

# ─── Delete operations ───────────────────────────────────────────────────────

KhepriBase.b_delete_all_shape_refs(b::GL) =
  begin
    clear_scene!(b.scene)
    nothing
  end

# ─── Connection lifecycle ─────────────────────────────────────────────────────

KhepriBase.after_connecting(b::GL) = nothing

# ─── GPU upload ───────────────────────────────────────────────────────────────

function upload_buffers(b::GL)
  GLFW.MakeContextCurrent(b.window)
  upload_primitive_buffers!(b.scene.tris)
  upload_primitive_buffers!(b.scene.lines)
  upload_primitive_buffers!(b.scene.points)
end

function upload_primitive_buffers!(pb::PrimitiveBuffers)
  isempty(pb.to_delete) && isempty(pb.pending) && !pb.dirty && return
  # Delete old GPU resources
  for chunk in pb.to_delete
    glDeleteVertexArrays(1, Ref(chunk.vao))
    glDeleteBuffers(1, Ref(chunk.vbo))
  end
  empty!(pb.to_delete)
  # Upload pending full buffers as new GPU chunks
  for (data, nv) in pb.pending
    vao = create_vao()
    vbo = create_vbo()
    setup_interleaved_vao!(vao, vbo)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
    push!(pb.chunks, GPUChunk(vao, vbo, nv, data))
  end
  empty!(pb.pending)
  # Upload current partial buffer
  if pb.dirty && pb.offset > 0
    glBindBuffer(GL_ARRAY_BUFFER, pb.current_vbo)
    glBufferData(GL_ARRAY_BUFFER, pb.offset * sizeof(Float32), pb.buffer, GL_STREAM_DRAW)
  end
  glBindBuffer(GL_ARRAY_BUFFER, 0)
  pb.dirty = false
end

# ─── Uniform setting ─────────────────────────────────────────────────────────

function set_uniforms(program, model, view_mat, proj, cam_pos)
  glUseProgram(program)
  loc = glGetUniformLocation(program, "uModel")
  loc >= 0 && glUniformMatrix4fv(loc, 1, GL_FALSE, model)
  loc = glGetUniformLocation(program, "uView")
  loc >= 0 && glUniformMatrix4fv(loc, 1, GL_FALSE, view_mat)
  loc = glGetUniformLocation(program, "uProjection")
  loc >= 0 && glUniformMatrix4fv(loc, 1, GL_FALSE, proj)
  loc = glGetUniformLocation(program, "uCameraPos")
  loc >= 0 && glUniform3f(loc, cam_pos[1], cam_pos[2], cam_pos[3])
end

# ─── Render pipeline ─────────────────────────────────────────────────────────

function compute_matrices(b::GL)
  let cam_pos = camera_position(b),
      target = b.cam_target,
      aspect = Float32(b.width) / Float32(b.height),
      view_mat = look_at_matrix(cam_pos, target, [0.0, 0.0, 1.0]),
      proj = perspective_matrix(45.0, aspect, 0.1, 10000.0),
      mdl = model_matrix()
    (mdl, view_mat, proj, Float32.(cam_pos))
  end
end

function draw_primitives(pb::PrimitiveBuffers, mode::GLenum)
  for chunk in pb.chunks
    glBindVertexArray(chunk.vao)
    glDrawArrays(mode, 0, chunk.vertex_count)
  end
  let nv = pb.offset ÷ VERTEX_FLOATS
    if nv > 0
      glBindVertexArray(pb.current_vao)
      glDrawArrays(mode, 0, nv)
    end
  end
end

function render_frame(b::GL)
  ensure_gpu(b)
  upload_buffers(b)

  # Resize window if render_size changed
  let rw = render_width(), rh = render_height()
    if rw != b.applied_render_width || rh != b.applied_render_height
      GLFW.SetWindowSize(b.window, rw, rh)
      b.applied_render_width = rw
      b.applied_render_height = rh
    end
  end

  # Get framebuffer size (handles HiDPI)
  w, h = GLFW.GetFramebufferSize(b.window)
  glViewport(0, 0, w, h)
  b.width = max(1, w)
  b.height = max(1, h)

  let mode = b.display_mode,
      bg = mode == :arctic ? (0.98f0, 0.98f0, 1.0f0, 1.0f0) : b.background_color
    glClearColor(bg...)
  end
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

  mdl, view_mat, proj, cam_pos = compute_matrices(b)

  let mode = b.display_mode
    # Draw solid triangles (skip for pure wireframe mode)
    if has_data(b.scene.tris) && mode != :wireframe
      let shader = if mode == :arctic
                     b.arctic_shader_program
                   elseif mode == :xray
                     b.xray_shader_program
                   else
                     b.shader_program
                   end
        set_uniforms(shader, mdl, view_mat, proj, cam_pos)
        draw_primitives(b.scene.tris, GL_TRIANGLES)
      end

      # Wireframe overlay for :shaded_wireframe mode
      if mode == :shaded_wireframe
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glDepthFunc(GL_LEQUAL)
        set_uniforms(b.flat_shader_program, mdl, view_mat, proj, cam_pos)
        draw_primitives(b.scene.tris, GL_TRIANGLES)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDepthFunc(GL_LESS)
      end
    end

    # Wireframe-only mode: draw triangle edges in GL_LINE mode
    if has_data(b.scene.tris) && mode == :wireframe
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
      set_uniforms(b.flat_shader_program, mdl, view_mat, proj, cam_pos)
      draw_primitives(b.scene.tris, GL_TRIANGLES)
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    end
  end

  # Draw lines with flat shading
  if has_data(b.scene.lines)
    set_uniforms(b.flat_shader_program, mdl, view_mat, proj, cam_pos)
    draw_primitives(b.scene.lines, GL_LINES)
  end

  # Draw points with flat shading
  if has_data(b.scene.points)
    set_uniforms(b.flat_shader_program, mdl, view_mat, proj, cam_pos)
    draw_primitives(b.scene.points, GL_POINTS)
  end

  glBindVertexArray(0)
  GLFW.SwapBuffers(b.window)
end

# ─── View and rendering ──────────────────────────────────────────────────────

function sync_camera_from_view(b::GL)
  let cam = in_world(b.view.camera),
      tgt = in_world(b.view.target),
      dx = cam.x - tgt.x,
      dy = cam.y - tgt.y,
      dz = cam.z - tgt.z,
      dist = sqrt(dx*dx + dy*dy + dz*dz)
    b.cam_target .= [tgt.x, tgt.y, tgt.z]
    b.cam_distance = max(0.1, dist)
    b.cam_azimuth = atan(dy, dx)
    b.cam_elevation = asin(clamp(dz / max(dist, 1e-10), -1.0, 1.0))
  end
end

KhepriBase.b_render_and_save_view(b::GL, path) =
  let win = ensure_window(b),
      was_visible = Bool(GLFW.GetWindowAttrib(win, GLFW.VISIBLE)),
      rw = render_width(),
      rh = render_height()
    GLFW.SetWindowSize(win, rw, rh)
    b.applied_render_width = rw
    b.applied_render_height = rh
    GLFW.ShowWindow(win)
    raise_window(win)
    sync_camera_from_view(b)
    render_frame(b)
    let w = b.width,
        h = b.height,
        pixels = Vector{UInt8}(undef, w * h * 3)
      glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels)
      save_png(path, w, h, pixels)
    end
    was_visible || GLFW.HideWindow(win)
    path
  end

KhepriBase.b_render_pathname(b::GL, name) =
  render_default_pathname(name) |> p -> replace(p, r"\.[^.]*$" => ".png")

function save_png(path, w, h, pixels)
  let png_path = endswith(path, ".png") ? path : replace(path, r"\.[^.]*$" => ".png"),
      # Reshape raw RGB bytes into (w, h) matrix of RGB values, flipping vertically
      img = Array{RGB{N0f8}}(undef, h, w)
    for row in 1:h
      let src_row = h - row + 1,  # flip vertically (OpenGL is bottom-up)
          offset = (src_row - 1) * w * 3
        for col in 1:w
          let i = offset + (col - 1) * 3
            img[row, col] = RGB{N0f8}(
              reinterpret(N0f8, pixels[i + 1]),
              reinterpret(N0f8, pixels[i + 2]),
              reinterpret(N0f8, pixels[i + 3]))
          end
        end
      end
    end
    PNGFiles.save(png_path, img)
    png_path
  end
end

# ─── Display and interaction ──────────────────────────────────────────────────

export open_view, close_view, display_view, set_wireframe, set_display_mode, set_msaa

"""
    set_wireframe(on::Bool, b=gl)

Toggle wireframe overlay on/off. When enabled, triangle edges are drawn
on top of the solid geometry.
"""
set_wireframe(on::Bool, b::GL=gl) = (b.display_mode = on ? :shaded_wireframe : :shaded; nothing)

"""
    set_display_mode(mode::Symbol, b=gl)

Set the display mode. Supported modes:
- `:shaded` — Phong shading (default)
- `:shaded_wireframe` — Phong shading with wireframe overlay
- `:wireframe` — Wireframe only
- `:arctic` — White material with soft multi-directional lighting
- `:xray` — Semi-transparent with edge enhancement
"""
set_display_mode(mode::Symbol, b::GL=gl) = (b.display_mode = mode; nothing)

"""
    set_msaa(samples::Int, b=gl)

Set MSAA anti-aliasing sample count. Use 0 to disable, 2/4/8 for increasing
quality. Default is 4. Takes effect on the next `open_view`/`display_view`
call (requires window recreation).
"""
function set_msaa(samples::Int, b::GL=gl)
  b.msaa_samples = samples
  # Force window recreation on next open
  if b.window !== nothing
    close_view(b)
    GLFW.DestroyWindow(b.window)
    b.window = nothing
    b.initialized = false
  end
  nothing
end

"""
    open_view(b=gl)

Open the GL window and start a background render loop.
The REPL remains interactive — shapes added afterwards appear live.
Close with `close_view()` or press Escape in the window.
"""
function open_view(b::GL=gl)
  # Stop any existing timer
  if b.render_timer !== nothing
    close(b.render_timer)
    b.render_timer = nothing
  end
  let win = ensure_window(b),
      rw = render_width(),
      rh = render_height()
    GLFW.SetWindowSize(win, rw, rh)
    b.applied_render_width = rw
    b.applied_render_height = rh
    GLFW.ShowWindow(win)
    raise_window(win)
    sync_camera_from_view(b)
    b.render_timer = Timer(0.0; interval=1/60) do _
      try
        if b.needs_restart
          b.needs_restart = false
          # Recreate window in-place (e.g. MSAA changed), keeping the same timer
          GLFW.DestroyWindow(b.window)
          b.window = nothing
          b.initialized = false
          let win = ensure_window(b),
              rw = render_width(),
              rh = render_height()
            GLFW.SetWindowSize(win, rw, rh)
            b.applied_render_width = rw
            b.applied_render_height = rh
            GLFW.ShowWindow(win)
            raise_window(win)
          end
        elseif b.window !== nothing && !GLFW.WindowShouldClose(b.window)
          render_frame(b)
          GLFW.PollEvents()
        else
          close_view(b)
        end
      catch e
        @warn "KhepriGL render error" exception=(e, catch_backtrace())
        close_view(b)
      end
    end
  end
  nothing
end

"""
    close_view(b=gl)

Stop the background render loop and hide the window.
"""
function close_view(b::GL=gl)
  if b.render_timer !== nothing
    close(b.render_timer)
    b.render_timer = nothing
  end
  if b.window !== nothing
    GLFW.HideWindow(b.window)
    GLFW.SetWindowShouldClose(b.window, false)
  end
  nothing
end

"""
    display_view(b=gl)

Blocking render loop — opens the window and only returns when it is closed.
For interactive use, prefer `open_view()` instead.
"""
function display_view(b::GL=gl)
  # Stop background timer if running
  close_view(b)
  let win = ensure_window(b),
      rw = render_width(),
      rh = render_height()
    GLFW.SetWindowSize(win, rw, rh)
    b.applied_render_width = rw
    b.applied_render_height = rh
    GLFW.ShowWindow(win)
    raise_window(win)
    sync_camera_from_view(b)
    while !GLFW.WindowShouldClose(win)
      render_frame(b)
      GLFW.PollEvents()
      sleep(1/60)
    end
    GLFW.HideWindow(win)
    GLFW.SetWindowShouldClose(win, false)
  end
  nothing
end

function collect_positions!(xs, ys, zs, pb::PrimitiveBuffers)
  # Collect positions from completed chunks
  for chunk in pb.chunks
    let data = chunk.data, n = chunk.vertex_count
      for v in 0:n-1
        let off = v * VERTEX_FLOATS
          push!(xs, data[off+1])
          push!(ys, data[off+2])
          push!(zs, data[off+3])
        end
      end
    end
  end
  # Collect positions from pending buffers
  for (data, nv) in pb.pending
    for v in 0:nv-1
      let off = v * VERTEX_FLOATS
        push!(xs, data[off+1])
        push!(ys, data[off+2])
        push!(zs, data[off+3])
      end
    end
  end
  # Collect positions from current partial buffer
  let nv = pb.offset ÷ VERTEX_FLOATS
    for v in 0:nv-1
      let off = v * VERTEX_FLOATS
        push!(xs, pb.buffer[off+1])
        push!(ys, pb.buffer[off+2])
        push!(zs, pb.buffer[off+3])
      end
    end
  end
end

KhepriBase.b_zoom_extents(b::GL) =
  let scene = b.scene
    if has_data(scene.tris) || has_data(scene.lines) || has_data(scene.points)
      let xs = Float32[], ys = Float32[], zs = Float32[]
        collect_positions!(xs, ys, zs, scene.tris)
        collect_positions!(xs, ys, zs, scene.lines)
        collect_positions!(xs, ys, zs, scene.points)
        if !isempty(xs)
          let cx = (minimum(xs) + maximum(xs)) / 2,
              cy = (minimum(ys) + maximum(ys)) / 2,
              cz = (minimum(zs) + maximum(zs)) / 2,
              dx = maximum(xs) - minimum(xs),
              dy = maximum(ys) - minimum(ys),
              dz = maximum(zs) - minimum(zs),
              radius = sqrt(dx*dx + dy*dy + dz*dz) / 2
            b.cam_target .= [cx, cy, cz]
            b.cam_distance = max(0.1, radius * 2.5)
          end
        end
      end
    end
  end

KhepriBase.b_set_view(b::GL, camera, target, lens, aperture) =
  begin
    b.view.camera = camera
    b.view.target = target
    b.view.lens = lens
    b.view.aperture = aperture
    sync_camera_from_view(b)
  end

KhepriBase.b_get_view(b::GL) =
  (b.view.camera, b.view.target, b.view.lens)
