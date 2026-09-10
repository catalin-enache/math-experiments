import math

import numpy as np

from imgui_bundle import imgui, hello_imgui

from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_LINES,
    GL_LINK_STATUS,
    GL_STATIC_DRAW,
    GL_TRUE,
    GL_UNSIGNED_INT,
    glAttachShader,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glClear,
    glClearColor,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteShader,
    glDisable,
    glDrawElements,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetAttribLocation,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
    GL_FRAGMENT_SHADER,
    GL_VERTEX_SHADER,
)


angle_normalized = 0.0

shader_program = None
vertex_array = None
vertex_buffer = None
index_buffer = None
mvp_location = None
color_location = None


VERTEX_SHADER_SOURCE = """
#version 150 core

in vec3 position;

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(position, 1.0);
}
"""


FRAGMENT_SHADER_SOURCE = """
#version 150 core

uniform vec3 line_color;

out vec4 fragment_color;

void main()
{
    fragment_color = vec4(line_color, 1.0);
}
"""


# Eight corners of a cube.
CUBE_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5],  # 7
    ],
    dtype=np.float32,
)


# Twelve cube edges, represented as pairs of vertex indices.
CUBE_INDICES = np.array(
    [
        # Back face
        0, 1,
        1, 2,
        2, 3,
        3, 0,

        # Front face
        4, 5,
        5, 6,
        6, 7,
        7, 4,

        # Connections between faces
        0, 4,
        1, 5,
        2, 6,
        3, 7,
    ],
    dtype=np.uint32,
)


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    success = glGetShaderiv(shader, GL_COMPILE_STATUS)

    if not success:
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compilation failed:\n{error}")

    return shader


def create_shader_program():
    vertex_shader = compile_shader(
        VERTEX_SHADER_SOURCE,
        GL_VERTEX_SHADER,
    )

    fragment_shader = compile_shader(
        FRAGMENT_SHADER_SOURCE,
        GL_FRAGMENT_SHADER,
    )

    program = glCreateProgram()

    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    success = glGetProgramiv(program, GL_LINK_STATUS)

    if not success:
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Shader linking failed:\n{error}")

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


def create_perspective_matrix(
    field_of_view_degrees,
    aspect_ratio,
    near_plane,
    far_plane,
):
    fov_radians = math.radians(field_of_view_degrees)
    f = 1.0 / math.tan(fov_radians / 2.0)

    matrix = np.zeros((4, 4), dtype=np.float32)

    matrix[0, 0] = f / aspect_ratio
    matrix[1, 1] = f

    matrix[2, 2] = (
        far_plane + near_plane
    ) / (
        near_plane - far_plane
    )

    matrix[2, 3] = (
        2.0 * far_plane * near_plane
    ) / (
        near_plane - far_plane
    )

    matrix[3, 2] = -1.0

    return matrix


def create_translation_matrix(x, y, z):
    matrix = np.identity(4, dtype=np.float32)

    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    return matrix


def create_axis_rotation_matrix(
    angle_degrees,
    axis_x,
    axis_y,
    axis_z,
):
    axis = np.array(
        [axis_x, axis_y, axis_z],
        dtype=np.float32,
    )

    axis_length = np.linalg.norm(axis)

    if axis_length == 0:
        return np.identity(4, dtype=np.float32)

    axis /= axis_length

    x, y, z = axis

    angle_radians = math.radians(angle_degrees)

    c = math.cos(angle_radians)
    s = math.sin(angle_radians)
    t = 1.0 - c

    matrix = np.array(
        [
            [
                t * x * x + c,
                t * x * y - s * z,
                t * x * z + s * y,
                0.0,
            ],
            [
                t * x * y + s * z,
                t * y * y + c,
                t * y * z - s * x,
                0.0,
            ],
            [
                t * x * z - s * y,
                t * y * z + s * x,
                t * z * z + c,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return matrix


def initialise():
    global shader_program
    global vertex_array
    global vertex_buffer
    global index_buffer
    global mvp_location
    global color_location

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.08, 1.0)

    shader_program = create_shader_program()

    vertex_array = glGenVertexArrays(1)
    glBindVertexArray(vertex_array)

    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)

    glBufferData(
        GL_ARRAY_BUFFER,
        CUBE_VERTICES.nbytes,
        CUBE_VERTICES,
        GL_STATIC_DRAW,
    )

    index_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)

    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        CUBE_INDICES.nbytes,
        CUBE_INDICES,
        GL_STATIC_DRAW,
    )

    position_location = glGetAttribLocation(
        shader_program,
        "position",
    )

    glEnableVertexAttribArray(position_location)

    glVertexAttribPointer(
        position_location,
        3,
        GL_FLOAT,
        GL_FALSE,
        3 * CUBE_VERTICES.itemsize,
        None,
    )

    mvp_location = glGetUniformLocation(
        shader_program,
        "mvp",
    )

    color_location = glGetUniformLocation(
        shader_program,
        "line_color",
    )

    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def render_scene():
    io = imgui.get_io()

    framebuffer_width = int(
        io.display_size.x *
        io.display_framebuffer_scale.x
    )

    framebuffer_height = int(
        io.display_size.y *
        io.display_framebuffer_scale.y
    )

    if framebuffer_width <= 0 or framebuffer_height <= 0:
        return

    glViewport(
        0,
        0,
        framebuffer_width,
        framebuffer_height,
    )

    glClearColor(0.05, 0.05, 0.08, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    aspect_ratio = (
        framebuffer_width /
        framebuffer_height
    )

    projection = create_perspective_matrix(
        field_of_view_degrees=60.0,
        aspect_ratio=aspect_ratio,
        near_plane=0.1,
        far_plane=100.0,
    )

    translation = create_translation_matrix(
        0.0,
        0.0,
        -3.0,
    )

    rotation = create_axis_rotation_matrix(
        angle_degrees=angle_normalized * 360.0,
        axis_x=10.0,
        axis_y=0.0,
        axis_z=1.0,
    )

    model_view = translation @ rotation
    mvp = projection @ model_view

    glEnable(GL_DEPTH_TEST)
    glUseProgram(shader_program)

    # NumPy stores the matrix in row-major order, so GL_TRUE asks
    # OpenGL to transpose it when uploading.
    glUniformMatrix4fv(
        mvp_location,
        1,
        GL_TRUE,
        mvp,
    )

    glUniform3f(
        color_location,
        1.0,
        1.0,
        1.0,
    )

    glBindVertexArray(vertex_array)

    glDrawElements(
        GL_LINES,
        len(CUBE_INDICES),
        GL_UNSIGNED_INT,
        None,
    )

    glBindVertexArray(0)
    glUseProgram(0)

    # Leave depth testing disabled for the ImGui overlay.
    glDisable(GL_DEPTH_TEST)


def render_gui():
    global angle_normalized

    imgui.set_next_window_pos(
        (15, 15),
        imgui.Cond_.once,
    )

    imgui.set_next_window_size(
        (320, 145),
        imgui.Cond_.once,
    )

    imgui.begin("Cube controls")

    changed, angle_normalized = imgui.slider_float(
        "Angle",
        angle_normalized,
        0.0,
        1.0,
        "%.3f",
    )

    imgui.separator()

    imgui.text(
        f"Normalized: {angle_normalized:.3f}"
    )

    imgui.text(
        f"Degrees: {angle_normalized * 360.0:.1f}"
    )

    if imgui.button("Reset"):
        angle_normalized = 0.0

    imgui.same_line()

    if imgui.button("90 degrees"):
        angle_normalized = 0.25

    imgui.end()


def main():
    runner_params = hello_imgui.RunnerParams()

    runner_params.app_window_params.window_title = (
        "PyOpenGL + ImGui Bundle"
    )

    runner_params.app_window_params.window_geometry.size = (
        1000,
        800,
    )

    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )

    runner_params.renderer_backend_type = (
        hello_imgui.RendererBackendType.open_gl3
    )

    runner_params.callbacks.post_init = initialise
    runner_params.callbacks.custom_background = render_scene
    runner_params.callbacks.show_gui = render_gui

    runner_params.ini_disable = True

    hello_imgui.run(runner_params)


if __name__ == "__main__":
    main()