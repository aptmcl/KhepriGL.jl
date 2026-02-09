```@meta
CurrentModule = KhepriGL
```

# KhepriGL

A pure-Julia OpenGL backend for Khepri, providing interactive 3D visualization via GLFW and ModernGL.

## Architecture

KhepriGL is a **local Julia backend** (not a socket or IO backend). It accumulates all geometry into flat interleaved vertex buffers (position + normal + color = 10 floats per vertex), then uploads to the GPU in batched chunks at render time.

- **Rendering pipeline**: OpenGL 3.3 core profile with GLSL shaders (Phong, flat, arctic, xray)
- **Buffer strategy**: Fixed-size CPU buffers flushed as GPU chunks (VBO+VAO), ~2.5 MB each
- **Coordinate system**: Right-handed Z-up (same as Khepri, no transforms needed)

## Key Features

- **Multiple display modes**: `:shaded`, `:wireframe`, `:arctic`, `:xray`, `:shaded_wireframe`
- **MSAA anti-aliasing**: Configurable sample count (0, 2, 4, or 8)
- **Interactive camera**: Orbit and pan via mouse, with configurable azimuth/elevation/distance
- **Dynamic lighting**: Up to 16 point or spot lights with inverse-square attenuation
- **Two-sided shading**: Hardcoded fallback lighting when no user lights are defined

## Setup

```julia
using KhepriGL
using KhepriBase

backend(gl)

# Create geometry
sphere(xyz(0, 0, 0), 5)
box(xyz(10, 0, 0), 5, 5, 5)
```

KhepriGL requires a display server (X11, Wayland, or Windows desktop). It will not work in headless environments without a virtual framebuffer.

## Dependencies

- **KhepriBase**: Core Khepri functionality
- **ModernGL**: OpenGL 3.3+ bindings
- **GLFW**: Window and input management
- **ColorTypes**: Color handling

```@index
```

```@autodocs
Modules = [KhepriGL]
```
