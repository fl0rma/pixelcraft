#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import pandas as pd

# ### 1. Inputs

def processing_image(uploaded_file, desired_pattern, count):
   
# ### 2. Cartoon Effect
    # def read_file(filename):
    #     img = cv2.imread(filename)
    #     cv2_imshow(img)
    #     return img
    # BORRAR
    # print(uploaded_file)
    # img = cv2.imread(uploaded_file)  # Read the image from the inputs
    img = uploaded_file  # Read the image from the inputs

    #Detect the edge in an image by using the cv2.adaptiveThreshold() function for a cartoon effect.
    def edge_mask(img, line_size, blur_value):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, blur_value)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
        return edges

    line_size = 7     #A larger line size means the thicker edges that will be emphasized in the image
    blur_value = 7    #The larger blur value means fewer black noises appear in the image

    edges = edge_mask(img, line_size, blur_value)

    # ##### Color Quantization

    def color_quantization(img, k):
         # Transform the image into a NumPy array of floating-point numbers, containing the RGB codes 
        data = np.float32(img).reshape((-1, 3))

        # Determine criteria: 20 max iterations, 0.001 epsilon value for level of error
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

        # Implementing K-Means (None for initial cluster centers, it will start randomly / 5 is the number of times it will run and it will use the best of those) 
        ret, label, center = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

        # Cluster centers are converted to unsigned 8-bit integers (uint8) to make them suitable for representing colors in images
        center = np.uint8(center)

        # Assigns each pixel in the image to its nearest cluster center, effectively recoloring the image
        result = center[label.flatten()]

        # Result array is converted back to the original shape of the input image (img)
        result = result.reshape(img.shape)
        return result

    # Apply color quantization 
    total_colors = 10
    img = color_quantization(img, total_colors)

    # ##### Bilateral Filter

    #we can reduce the noise in the image by using a bilateral filter. It would give a bit blurred and sharpness-reducing effect to the image.
    d = 7            #Diameter of each pixel neighborhood
    sigmaColor=200   #A larger value of the parameter means larger areas of semi-equal color.
    sigmaSpace=200  #A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough.

    blurred = cv2.bilateralFilter(img, d, sigmaColor,sigmaSpace)

    # ##### Combine Edge Mask with the Colored Image

    # Combining the edge mask that we created earlier, with the color-processed image.
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

# ### 3. Pattern processing
    new_width  = 1000
    pixel_size = int(new_width / int(count))

    # Get the current height and width of the image
    height, width, _ = cartoon.shape

    # Calculate new height while maintaining aspect ratio
    new_height = int(new_width * height / width)

    # Resize the image using OpenCV
    resized_img = cv2.resize(cartoon, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    def pixelate(image_array, pixel_size):
        # Extracts the height and width using the shape attribute
        img_height, img_width, _ = image_array.shape
    
        # Calculates the number of steps or blocks along the x and y dimensions of the image based on the pixel_size. 
        x_steps = img_width // pixel_size
        y_steps = img_height // pixel_size
    
        # This array will be used to store the pixelated version of the image.
        pixelated_image = np.zeros_like(image_array)

        for y in range(y_steps):          # This outer loop iterates over the vertical blocks (rows) of the image.
            for x in range(x_steps):      # This inner loop iterates over the horizontal blocks (columns) of the image.
                # This block is essentially a rectangular region of the image, and its size is determined by the pixel_size variable
                block = image_array[y * pixel_size : (y + 1) * pixel_size, x * pixel_size : (x + 1) * pixel_size]
                # Within each block, the code calculates the average color.
                average_color = np.mean(np.mean(block, axis=0), axis=0)
                # The average color calculated for each block is assigned to all the pixels within that block in the pixelated_image.
                pixelated_image[y * pixel_size : (y + 1) * pixel_size, x * pixel_size : (x + 1) * pixel_size] = average_color

        return pixelated_image

    # Call the pixelate function with the calculated pixel size
    pixelated_image = pixelate(resized_img, pixel_size)

    # Specific list of RGB colors, representing the available color options for the 1X1 Plate piece at the Lego store and closest thread color
    lego_data = pd.read_csv('./rgb_lego_colors.csv')
    thread_data = pd.read_csv('./rgb_threads_colors.csv', encoding='ISO-8859-1')

    # Convert DataFrame to a list of RGB tuples
    lego_rgb_values = [(r, g, b) for r, g, b in zip(lego_data['R'], lego_data['G'], lego_data['B'])]
    thread_rgb_values = [(r, g, b) for r, g, b in zip(thread_data['R'], thread_data['G'], thread_data['B'])]

    def color_quantization2(img, colors):
        data = np.float32(img).reshape((-1, 3))
    
        #Diference respect to color_quantization is that this starts with the specific colors indicated as the centers of the clusters
        centers = np.array(colors, dtype=np.float32)
    
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, label, center = cv2.kmeans(data, len(colors), None, criteria, 10, cv2.KMEANS_USE_INITIAL_LABELS, centers)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(img.shape)
        return result

    # Apply color quantization with the specific colors, to ensure the final result has the avaliable colors
    if desired_pattern == "lego":
        specific_colors = lego_rgb_values
    else:
        specific_colors = thread_rgb_values

    final_image = color_quantization2(pixelated_image, specific_colors)   

    # Convert the image to the BGR color space
    bgr_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

    # Create an empty dictionary to store square colors and counts
    square_colors = {tuple(color): 0 for color in specific_colors}

    # Function to find the closest color in the specific_colors list
    def find_closest_color(color):
        color = np.array(color)
        distances = np.linalg.norm(color - specific_colors, axis=1)
        closest_color_index = np.argmin(distances)
        return tuple(specific_colors[closest_color_index])

    # Determine the size of each square based on the gridline spacing
    square_size = pixel_size

    # Create a copy of the image to modify
    result_image = bgr_image.copy()

    # Create a dictionary to store color-to-symbol mappings
    color_to_symbol = {}

    # Generate a list of unique symbols
    unique_symbols = list(range(1, 39+1))  

    # Iterate through the unique colors and assign a symbol to each
    for i, (color, _) in enumerate(square_colors.items()):
        symbol = unique_symbols[i]  # Get the next symbol from the list
        color_to_symbol[color] = symbol

    # Use color_to_symbol to map colors to symbols in result_image
    for y in range(0, bgr_image.shape[0], square_size):
        for x in range(0, bgr_image.shape[1], square_size):
            square = bgr_image[y:y+square_size, x:x+square_size]
            square_color = tuple(np.mean(square, axis=(0, 1), dtype=int))
            closest_color = find_closest_color(square_color)
        
            # Get the symbol for the closest color from the dictionary
            symbol = color_to_symbol.get(closest_color, 'N/A')  # Add a default value for debugging
              
            # Increment the count of the closest color in square_colors
            square_colors[closest_color] += 1
        
            # Replace the colors in the square with the closest color
            result_image[y:y+square_size, x:x+square_size] = closest_color
        
            # Overlay the symbol on top of the square
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = (pixel_size / 50)
            font_color = (0, 0, 0)  # Black font color
            font_thickness = 1
              
            # Calculate the text size to determine the width and height of the text
            (text_width, text_height), _ = cv2.getTextSize(str(symbol), font, font_scale, font_thickness)

            # Calculate the position (org) to center the text within the square
            x_centered = x + (square_size - text_width) // 2
            y_centered = y + (square_size + text_height) // 2
            org = (x_centered, y_centered)

            # Overlay the symbol on top of the square
            cv2.putText(result_image, str(symbol), org, font, font_scale, font_color, font_thickness)

    # Convert the modified image back to RGB color space
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Determine the gridline spacing based on pixel size
    gridline_spacing = pixel_size

    # Create a copy of the pixelated image to draw gridlines on
    image_with_gridlines = result_image_rgb.copy()

    # Draw vertical gridlines -  iterates over the coordinates of the image and draws black lines 
    for x in range(0, pixelated_image.shape[1], gridline_spacing):
        cv2.line(image_with_gridlines, (x, 0), (x, pixelated_image.shape[0]), (0, 0, 0), 1)  

    # Draw horizontal gridlines  -  iterates over the coordinates of the image and draws black lines 
    for y in range(0, pixelated_image.shape[0], gridline_spacing):
        cv2.line(image_with_gridlines, (0, y), (pixelated_image.shape[1], y), (0, 0, 0), 1)  

    #cv2.imwrite('image_with_gridlines.png', image_with_gridlines)

# ### 4. Prepare legend

    # Convert the dictionary to a DataFrame
    colors_total = pd.DataFrame(list(square_colors.items()), columns=["color", "count"])
    symbols_total = pd.DataFrame(list(color_to_symbol.items()), columns=["color", "symbol"])

    colors_used = pd.merge(colors_total, symbols_total, on='color', how='left')

    # Drop rows where Count is zero
    colors_used = (colors_used[colors_used["count"] != 0]).reset_index(drop=True)

    def clean_color(x):
        return tuple(map(int, str(x).replace(' ', '').strip('()').split(',')))

    colors_used['color'] = colors_used['color'].apply(clean_color)
    lego_data['color'] = lego_data['color'].apply(clean_color)
    thread_data['color'] = thread_data['color'].apply(clean_color)

    if desired_pattern == "lego":
        color_table = pd.merge(colors_used, lego_data, left_on='color', right_on = 'color', how='left')
    else:
        color_table = pd.merge(colors_used, thread_data, left_on='color', right_on = 'color', how='left')

    columns_to_drop = ['color','R', 'G','B',]
    color_table.drop(columns_to_drop, axis=1, inplace=True)

    data = {'image_with_gridlines': image_with_gridlines, 'color_table':color_table}

    return data
