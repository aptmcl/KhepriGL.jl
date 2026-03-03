# KhepriGL tests — Pure-Julia OpenGL backend
#
# These tests cover module loading, type system, backend struct defaults,
# scene buffer structures, and pure-Julia helpers. GPU/GLFW operations
# are not tested (they require a display server).

using KhepriGL
using KhepriBase
using Test

@testset "KhepriGL.jl" begin

  @testset "Type system" begin
    @test isdefined(KhepriGL, :GLKey)
    @test KhepriGL.GLId === Int
    @test isdefined(KhepriGL, :GLRef)
    @test isdefined(KhepriGL, :GLNativeRef)
    @test KhepriGL.GL === KhepriGL.GLBackend
  end

  @testset "Backend initialization" begin
    @test gl isa KhepriBase.Backend
    @test KhepriBase.backend_name(gl) == "GL"
    @test KhepriBase.void_ref(gl) === 0
    @test KhepriBase.view_type(KhepriGL.GL) isa KhepriBase.FrontendView
  end

  @testset "Backend struct defaults" begin
    b = KhepriGL.GL()
    @test b.display_mode === :shaded
    @test b.msaa_samples == 4
    @test b.width == 1024
    @test b.height == 768
    @test b.initialized == false
    @test b.next_id == 1
    @test isnothing(b.window)
    @test hasproperty(b, :refs)
    @test b.refs isa KhepriBase.References
    @test hasproperty(b, :view)
    @test b.view isa KhepriBase.View
    @test b.scene isa KhepriGL.GLScene
    @test b.background_color == (0.9f0, 0.9f0, 0.92f0, 1.0f0)
  end

  @testset "Constants" begin
    @test KhepriGL.VERTEX_FLOATS == 10
    @test KhepriGL.CHUNK_MAX_VERTICES == 65_532
    @test KhepriGL.MAX_LIGHTS == 16
    @test KhepriGL.LIGHT_POINT === Int32(0)
    @test KhepriGL.LIGHT_SPOT === Int32(1)
  end

  @testset "GLScene and PrimitiveBuffers" begin
    scene = KhepriGL.GLScene()
    @test scene.tris isa KhepriGL.PrimitiveBuffers
    @test scene.lines isa KhepriGL.PrimitiveBuffers
    @test scene.points isa KhepriGL.PrimitiveBuffers
    @test scene.lights isa Vector{KhepriGL.GLLight}
    @test isempty(scene.lights)

    pb = KhepriGL.PrimitiveBuffers()
    @test isempty(pb.chunks)
    @test isempty(pb.to_delete)
    @test isempty(pb.pending)
    @test length(pb.buffer) == KhepriGL.CHUNK_MAX_VERTICES * KhepriGL.VERTEX_FLOATS
    @test pb.offset == 0
    @test pb.dirty == false
  end

  @testset "GPUChunk struct" begin
    @test fieldnames(KhepriGL.GPUChunk) == (:vao, :vbo, :vertex_count, :data)
  end

  @testset "GLLight struct" begin
    @test fieldnames(KhepriGL.GLLight) == (:type, :position, :direction, :color, :energy, :hotspot, :falloff)
  end

  @testset "Scene management" begin
    scene = KhepriGL.GLScene()
    @test !KhepriGL.has_data(scene.tris)
    @test !KhepriGL.has_data(scene.lines)

    KhepriGL.clear_scene!(scene)
    @test isempty(scene.lights)
    @test !KhepriGL.has_data(scene.tris)
  end
end
