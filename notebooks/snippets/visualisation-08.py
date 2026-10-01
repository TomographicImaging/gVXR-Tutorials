# Compute the angle in radian
theta = 30 * pi / 180

# Build the matrix
matrix = [cos(theta), sin(theta), 0,
   sin(theta), cos(theta), 0,
   0, 0, 1]

# Retrieve the current position
eye_position = gvxr.getCameraPosition("cm")

# Rotate the position
eye_X = eye_position[0] * matrix[0] + eye_position[1] * matrix[3] + eye_position[2] * matrix[6]
eye_Y = eye_position[0] * matrix[1] + eye_position[1] * matrix[4] + eye_position[2] * matrix[7]
eye_Z = eye_position[0] * matrix[2] + eye_position[1] * matrix[5] + eye_position[2] * matrix[8]

# Set the new position
gvxr.setCameraPosition(eye_X, eye_Y, eye_Z, "cm")

# Retrieve the current position
target_position = gvxr.getCameraReferencePoint("cm")

# Rotate the position
target_X = target_position[0] * matrix[0] + target_position[1] * matrix[3] + target_position[2] * matrix[6]
target_Y = target_position[0] * matrix[1] + target_position[1] * matrix[4] + target_position[2] * matrix[7]
target_Z = target_position[0] * matrix[2] + target_position[1] * matrix[5] + target_position[2] * matrix[8]

# Set the new position
gvxr.setCameraReferencePoint(target_X, target_Y, target_Z, "cm")

# Do not show the beam
gvxr.displayBeam(False)

# Update the visualisation
gvxr.showWindow()
gvxr.displayScene()

# Take a screenshot
screenshot = gvxr.takeScreenshot()
gvxr.hideWindow()

# Display it using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(screenshot)
plt.title("Screenshot of the X-ray simulation environment")
plt.axis('off')
plt.show()