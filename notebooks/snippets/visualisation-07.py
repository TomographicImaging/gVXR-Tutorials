# Change the opacity
gvxr.setZoom(200)

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