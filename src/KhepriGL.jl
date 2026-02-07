module KhepriGL
using KhepriBase
using GLFW
using ModernGL
using LinearAlgebra
using Dates
using PNGFiles
using ColorTypes: RGB, N0f8

# functions that need specialization
include(khepribase_interface_file())

include("GL.jl")

function __init__()
  add_current_backend(gl)
  # Register GL-specific colors for predefined materials
  set_material(GLBackend, material_basic,    rgba(0.7, 0.7, 0.7, 1))
  set_material(GLBackend, material_glass,    rgba(0.8, 0.9, 1.0, 0.3))
  set_material(GLBackend, material_metal,    rgba(0.8, 0.8, 0.85, 1))
  set_material(GLBackend, material_wood,     rgba(0.55, 0.35, 0.17, 1))
  set_material(GLBackend, material_concrete, rgba(0.65, 0.65, 0.63, 1))
  set_material(GLBackend, material_plaster,  rgba(0.9, 0.88, 0.85, 1))
  set_material(GLBackend, material_grass,    rgba(0.3, 0.55, 0.2, 1))
  set_material(GLBackend, material_clay,     rgba(0.76, 0.5, 0.33, 1))
  set_material(GLBackend, material_point,    rgba(1.0, 1.0, 1.0, 1))
  set_material(GLBackend, material_curve,    rgba(0.0, 0.0, 0.0, 1))
  set_material(GLBackend, material_surface,  rgba(0.6, 0.6, 0.6, 1))
  open_view(gl)
end

end
