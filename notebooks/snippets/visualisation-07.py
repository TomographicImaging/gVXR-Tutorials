# Change the positions
gvxr.setCameraPosition(3, 60.0, 0.0, "cm")
gvxr.setCameraReferencePoint(3, 0.0, 0.0, "cm")

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