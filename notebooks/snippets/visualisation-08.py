# Compute the angle in radian
theta = 30 * pi / 180

# 1. Build the rotation matrix
matrix = [cos(theta), sin(theta), 0,
   sin(theta), cos(theta), 0,
   0, 0, 1]

# 2. Retrieve the eye and reference point in two separate variables
eye_position = gvxr.getCameraPosition("cm")
target_position = gvxr.getCameraReferencePoint("cm")

# 3. Multiply the corresponding vectors with the rotation matrix
eye_X = eye_position[0] * matrix[0] + eye_position[1] * matrix[3] + eye_position[2] * matrix[6]
eye_Y = eye_position[0] * matrix[1] + eye_position[1] * matrix[4] + eye_position[2] * matrix[7]
eye_Z = eye_position[0] * matrix[2] + eye_position[1] * matrix[5] + eye_position[2] * matrix[8]

target_X = target_position[0] * matrix[0] + target_position[1] * matrix[3] + target_position[2] * matrix[6]
target_Y = target_position[0] * matrix[1] + target_position[1] * matrix[4] + target_position[2] * matrix[7]
target_Z = target_position[0] * matrix[2] + target_position[1] * matrix[5] + target_position[2] * matrix[8]

# 4 Set the new positions
gvxr.setCameraPosition(eye_X, eye_Y, eye_Z, "cm")
gvxr.setCameraReferencePoint(target_X, target_Y, target_Z, "cm")

# 5. Update the visualisation
gvxr.showWindow()
gvxr.displayScene()

# 6. Take a screenshot
screenshot = gvxr.takeScreenshot()
gvxr.hideWindow()

# 7. Display it using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(screenshot)
plt.title("Screenshot of the X-ray simulation environment")
plt.axis('off')
plt.show()