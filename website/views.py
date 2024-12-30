import os
from django.shortcuts import render
from django.http import HttpResponse
from .forms import PhotoUploadForm
from PIL import Image
from .Pneumonia_Model_script import runImageTesting

def testing(request):
    result_message = None  # Initialize result message

    if request.method == 'POST' and request.FILES.get('image'):
        form = PhotoUploadForm(request.POST, request.FILES)

        if form.is_valid():
            # Get the uploaded image from the request
            image_file = request.FILES['image']

            # Open the image using PIL (Python Imaging Library)
            image = Image.open(image_file)

            # Run the image through the model for prediction
            result = runImageTesting(image)

            # Create a response message with the result
            result_message = f"Image analyzed. Result: {result}% of pneumonia"

    else:
        form = PhotoUploadForm()

    return render(request, 'upload_photo.html', {'form': form, 'result_message': result_message})
