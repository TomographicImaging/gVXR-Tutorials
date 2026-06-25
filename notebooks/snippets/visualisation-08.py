# Change the transformation matrix
theta = 30 * pi / 180

matrix = [cos(theta), 0, theta, 0, \
   0, 1, 0, 0, \
   -sin(theta), 0, cos(theta), 0,
   0, 0, 0, 1]

gvxr.setSceneRotationMatrix(matrix)

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