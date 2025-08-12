import cv2
import matplotlib.pyplot as plt

def debug_frame(frame_masked, diff, thresh, contours):
    """Visualize intermediate processing results for debugging."""
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    axs[0].imshow(cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Masked Frame")

    axs[1].imshow(diff, cmap='gray')
    axs[1].set_title("Difference (Background Subtracted)")

    axs[2].imshow(thresh, cmap='gray')
    axs[2].set_title("Thresholded Image")

    contour_img = frame_masked.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    axs[3].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axs[3].set_title("Contours Found")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
