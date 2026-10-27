# 1. Show the normal vectors of the sample in the 3D visualisation
gvxr.displayNormalVectors(True)

# 2. Display the 3D objects in wireframe mode
gvxr.useWireframe(True)

# 3. Update the visualisationb
gvxr.showWindow()
gvxr.displayScene()

# 4. Take a screenshot
screenshot = gvxr.takeScreenshot()
gvxr.hideWindow()

# 5. Display it using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(screenshot)
plt.title("Screenshot of the X-ray simulation environment")
plt.axis('off')
plt.show()