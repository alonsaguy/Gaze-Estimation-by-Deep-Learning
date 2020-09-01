"""Utility methods for generating gazemaps."""
import cv2 as cv
import numpy as np
import tensorflow as tf

height_to_eyeball_radius_ratio = 1.1
eyeball_radius_to_iris_diameter_ratio = 1.0

def from_gaze2d(gaze, output_size, scale=1.0):
    """Generate a normalized pictorial representation of 3D gaze direction."""
    gazemaps = []
    oh, ow = np.round(scale * np.asarray(output_size)).astype(np.int32)
    oh_2 = int(np.round(0.5 * oh))
    ow_2 = int(np.round(0.5 * ow))
    r = int(height_to_eyeball_radius_ratio * oh_2)
    theta, phi = gaze
    theta = -theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Draw iris
    eyeball_radius = int(height_to_eyeball_radius_ratio * oh_2)
    iris_radius_angle = np.arcsin(0.5 * eyeball_radius_to_iris_diameter_ratio)
    iris_radius = eyeball_radius_to_iris_diameter_ratio * eyeball_radius
    iris_distance = float(eyeball_radius) * np.cos(iris_radius_angle)
    iris_offset = np.asarray([
        -iris_distance * sin_phi * cos_theta,
        iris_distance * sin_theta,
    ])
    iris_centre = np.asarray([ow_2, oh_2]) + iris_offset
    angle = np.degrees(np.arctan2(iris_offset[1], iris_offset[0]))
    ellipse_max = eyeball_radius_to_iris_diameter_ratio * iris_radius
    ellipse_min = np.abs(ellipse_max * cos_phi * cos_theta)
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv.ellipse(gazemap, box=(iris_centre, (ellipse_min, ellipse_max), angle),
                         color=1.0, thickness=-1, lineType=cv.LINE_AA)
    gazemaps.append(gazemap)

    # Draw eyeball
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    gazemap = cv.circle(gazemap, (ow_2, oh_2), r, color=1, thickness=-1)
    gazemaps.append(gazemap)

    return np.asarray(gazemaps)

"""Utility methods for generating and visualizing heatmaps."""
def gaussian_2d(shape, centre, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float32)
    ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float32), -1)
    alpha = -0.5 / (sigma**2)
    heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
    return heatmap

"""Utility methods for gaze angle and error calculations."""
def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

radians_to_degrees = 180.0 / np.pi


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees


def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return np.mean(angular_error(a, b))


def tensorflow_angular_error_from_pitchyaw(y_true, y_pred):
    """Tensorflow method to calculate angular loss from head angles."""
    def angles_to_unit_vectors(y):
        sin = tf.sin(y)
        cos = tf.cos(y)
        return tf.stack([
            tf.multiply(cos[:, 0], sin[:, 1]),
            sin[:, 0],
            tf.multiply(cos[:, 0], cos[:, 1]),
        ], axis=1)

    with tf.name_scope('mean_angular_error'):
        v_true = angles_to_unit_vectors(y_true)
        v_pred = angles_to_unit_vectors(y_pred)
        return tensorflow_angular_error_from_vector(v_true, v_pred)


def tensorflow_angular_error_from_vector(v_true, v_pred):
    """Tensorflow method to calculate angular loss from 3D vector."""
    with tf.name_scope('mean_angular_error'):
        v_true_norm = tf.sqrt(tf.reduce_sum(tf.square(v_true), axis=1))
        v_pred_norm = tf.sqrt(tf.reduce_sum(tf.square(v_pred), axis=1))

        sim = tf.div(tf.reduce_sum(tf.multiply(v_true, v_pred), axis=1),
                     tf.multiply(v_true_norm, v_pred_norm))

        # Floating point precision can cause sim values to be slightly outside of
        # [-1, 1] so we clip values
        sim = tf.clip_by_value(sim, -1.0 + 1e-6, 1.0 - 1e-6)

        ang = tf.scalar_mul(radians_to_degrees, tf.acos(sim))
        return tf.reduce_mean(ang)


def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out

def angle_to_screen(gaze_in_angles, position=[0,15,60], screen_size=[34.5, 19.5]): 
    """ Calculate the intersection point between the gaze vector and the screen """
    # screen_size = alonsag screen size in [cm]
    # position = position of the eye in [cm] relative to the camera
    gaze_vec = [-np.tan(gaze_in_angles[0]), -np.tan(gaze_in_angles[1]), -1.0]
    T = position[2]/gaze_vec[2]    
    return [position[0] + T*gaze_vec[0] + screen_size[0]/2, position[1] - T*gaze_vec[1]]