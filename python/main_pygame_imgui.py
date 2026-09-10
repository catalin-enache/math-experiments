import math

import numpy as np
import pygame

from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer
from OpenGL.GL import *


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

BACKGROUND_COLOR = (0.05, 0.05, 0.08, 1.0)
CUBE_COLOR = (1.0, 1.0, 1.0)

# 0.0 corresponds to 0 degrees.
# 1.0 corresponds to 360 degrees.
angle_normalized = 0.0


VERTEX_SHADER_SOURCE = """
#version 330 core

in vec3 position;

uniform mat4 mvp;

void main()
{
    gl_Position = mvp * vec4(position, 1.0);
}
"""


FRAGMENT_SHADER_SOURCE = """
#version 330 core

uniform vec3 line_color;

out vec4 fragment_color;

void main()
{
    fragment_color = vec4(line_color, 1.0);
}
"""


CUBE_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5],
    ],
    dtype=np.float32,
)


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

        # Connections between
        0, 4,
        1, 5,
        2, 6,
        3, 7,
    ],
    dtype=np.uint32,
)


shader_program = None
vertex_array = None
vertex_buffer = None
index_buffer = None
mvp_location = None
color_location = None


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)

    glShaderSource(shader, source)
    glCompileShader(shader)

    compilation_succeeded = glGetShaderiv(
        shader,
        GL_COMPILE_STATUS,
    )

    if not compilation_succeeded:
        error = glGetShaderInfoLog(shader).decode()

        raise RuntimeError(
            f"Shader compilation failed:\n{error}"
        )

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

    linking_succeeded = glGetProgramiv(
        program,
        GL_LINK_STATUS,
    )

    if not linking_succeeded:
        error = glGetProgramInfoLog(program).decode()

        raise RuntimeError(
            f"Shader linking failed:\n{error}"
        )

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program


def create_perspective_matrix(
    field_of_view_degrees,
    aspect_ratio,
    near_plane,
    far_plane,
):
    field_of_view_radians = math.radians(
        field_of_view_degrees
    )

    focal_length = 1.0 / math.tan(
        field_of_view_radians / 2.0
    )

    matrix = np.zeros(
        (4, 4),
        dtype=np.float32,
    )

    matrix[0, 0] = focal_length / aspect_ratio
    matrix[1, 1] = focal_length

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
    matrix = np.identity(
        4,
        dtype=np.float32,
    )

    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    return matrix


def create_y_rotation_matrix(angle_degrees):
    angle_radians = math.radians(
        angle_degrees
    )

    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)

    return np.array(
        [
            [ cosine, 0.0,  sine, 0.0],
            [    0.0, 1.0,   0.0, 0.0],
            [-sine,   0.0, cosine, 0.0],
            [    0.0, 0.0,   0.0, 1.0],
        ],
        dtype=np.float32,
    )


def initialise_cube():
    global shader_program
    global vertex_array
    global vertex_buffer
    global index_buffer
    global mvp_location
    global color_location

    shader_program = create_shader_program()

    vertex_array = glGenVertexArrays(1)
    glBindVertexArray(vertex_array)

    vertex_buffer = glGenBuffers(1)

    glBindBuffer(
        GL_ARRAY_BUFFER,
        vertex_buffer,
    )

    glBufferData(
        GL_ARRAY_BUFFER,
        CUBE_VERTICES.nbytes,
        CUBE_VERTICES,
        GL_STATIC_DRAW,
    )

    index_buffer = glGenBuffers(1)

    glBindBuffer(
        GL_ELEMENT_ARRAY_BUFFER,
        index_buffer,
    )

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

    glEnableVertexAttribArray(
        position_location
    )

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


def destroy_cube():
    if vertex_buffer is not None:
        glDeleteBuffers(1, [vertex_buffer])

    if index_buffer is not None:
        glDeleteBuffers(1, [index_buffer])

    if vertex_array is not None:
        glDeleteVertexArrays(1, [vertex_array])

    if shader_program is not None:
        glDeleteProgram(shader_program)


def render_cube(width, height):
    if width <= 0 or height <= 0:
        return

    aspect_ratio = width / height

    projection_matrix = create_perspective_matrix(
        field_of_view_degrees=60.0,
        aspect_ratio=aspect_ratio,
        near_plane=0.1,
        far_plane=100.0,
    )

    translation_matrix = create_translation_matrix(
        0.0,
        0.0,
        -3.0,
    )

    angle_degrees = (
        angle_normalized * 360.0
    )

    rotation_matrix = create_y_rotation_matrix(
        angle_degrees
    )

    model_view_matrix = (
        translation_matrix @ rotation_matrix
    )

    mvp_matrix = (
        projection_matrix @ model_view_matrix
    )

    glEnable(GL_DEPTH_TEST)
    glUseProgram(shader_program)

    # NumPy stores this matrix in row-major order.
    # GL_TRUE tells OpenGL to transpose it while uploading.
    glUniformMatrix4fv(
        mvp_location,
        1,
        GL_TRUE,
        mvp_matrix,
    )

    glUniform3f(
        color_location,
        *CUBE_COLOR,
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


def render_gui():
    global angle_normalized

    imgui.set_next_window_pos(
        (15, 15),
        imgui.Cond_.always,
    )

    imgui.set_next_window_size(
        (330, 140),
        imgui.Cond_.always,
    )

    imgui.begin(
        "Cube controls",
        flags=imgui.WindowFlags_.no_saved_settings,
    )

    changed, angle_normalized = imgui.slider_float(
        "Angle",
        angle_normalized,
        0.0,
        1.0,
        "%.3f",
    )

    imgui.text(f"Normalized: {angle_normalized:.3f}")
    imgui.text(f"Degrees: {angle_normalized * 360.0:.1f}")

    if imgui.button("Reset"):
        angle_normalized = 0.0

    imgui.same_line()

    if imgui.button("90 degrees"):
        angle_normalized = 0.25

    imgui.end()


def main():
    pygame.init()

    # Request a modern OpenGL context.
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_MAJOR_VERSION,
        3,
    )

    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_MINOR_VERSION,
        3,
    )

    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )

    # Required for a core OpenGL context on macOS.
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG,
        1,
    )

    pygame.display.set_mode(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        pygame.DOUBLEBUF |
        pygame.OPENGL,
    )

    pygame.display.set_caption(
        "Pygame + PyOpenGL + ImGui"
    )

    glEnable(GL_DEPTH_TEST)
    glClearColor(*BACKGROUND_COLOR)

    initialise_cube()

    # ImGui must be initialized after the OpenGL context exists.
    imgui.create_context()

    io = imgui.get_io()
    io.ini_saving_rate = float("inf")

    io.display_size = (
        float(SCREEN_WIDTH),
        float(SCREEN_HEIGHT),
    )

    imgui_renderer = PygameRenderer()

    clock = pygame.time.Clock()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            imgui_renderer.process_event(event)

        imgui_renderer.process_inputs()

        width, height = (
            pygame.display.get_window_size()
        )

        # Required before imgui.new_frame().
        io.display_size = (
            float(width),
            float(height),
        )

        imgui.new_frame()

        # Construct the GUI for this frame.
        render_gui()

        # Render the 3D scene.
        glViewport(0, 0, width, height)
        glClearColor(*BACKGROUND_COLOR)

        glClear(
            GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT
        )

        render_cube(width, height)

        # Render ImGui last so it appears over the cube.
        imgui.render()

        imgui_renderer.render(
            imgui.get_draw_data()
        )

        pygame.display.flip()
        clock.tick(60)

    imgui_renderer.shutdown()
    destroy_cube()

    pygame.quit()


if __name__ == "__main__":
    main()