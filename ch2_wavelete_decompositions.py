#Code
# Feature Extraction Chapter
# Example of successive wavelet decompositions. The $LL$ subband is iteratively decomposed, yielding finer subbands at multiple resolutions.

import matplotlib.pyplot as plt
	import pywt
	import pywt.data
	
	# Load grayscale image
	image = pywt.data.camera()
	
	# ---------------- Level 1 ----------------
	coeffs1 = pywt.dwt2(image, 'haar')
	LL1, (LH1, HL1, HH1) = coeffs1
	
	# ---------------- Level 2 ----------------
	coeffs2 = pywt.dwt2(LL1, 'haar')
	LL2, (LH2, HL2, HH2) = coeffs2
	
	# ---------------- Level 3 ----------------
	coeffs3 = pywt.dwt2(LL2, 'haar')
	LL3, (LH3, HL3, HH3) = coeffs3
	
	# --- Level 1 output ---
	arr1, _ = pywt.coeffs_to_array([LL1, (LH1, HL1, HH1)])
	
	# --- Level 2 output (LL1 replaced) ---
	arr2, _ = pywt.coeffs_to_array([LL2, (LH2, HL2, HH2)])
	arr_level2, _ = pywt.coeffs_to_array([arr2, (LH1, HL1, HH1)])
	
	# --- Level 3 output (LL2 replaced) ---
	arr3, _ = pywt.coeffs_to_array([LL3, (LH3, HL3, HH3)])
	arr_level3, _ = pywt.coeffs_to_array([arr3, (LH2, HL2, HH2)])
	arr_final, _ = pywt.coeffs_to_array([arr_level3, (LH1, HL1, HH1)])
	
	# ---------------- Visualization ----------------
	fig, axes = plt.subplots(2, 2, figsize=(12, 12))
	
	# Top-left: Original
	axes[0,0].imshow(image, cmap='gray')
	axes[0,0].set_title("Original Image")
	axes[0,0].axis('off')
	
	# Top-right: Level 1
	axes[0,1].imshow(arr1, cmap='gray')
	axes[0,1].set_title("Level 1")
	axes[0,1].axis('off')
	
	# Bottom-left: Level 2
	axes[1,0].imshow(arr_level2, cmap='gray')
	axes[1,0].set_title("Level 2")
	axes[1,0].axis('off')
	
	# Bottom-right: Level 3
	axes[1,1].imshow(arr_final, cmap='gray')
	axes[1,1].set_title("Level 3")
	axes[1,1].axis('off')
	
	plt.tight_layout()
	plt.show()
