from django.urls import path
from sendimage.views import get_image

urlpatterns = [
    path('getimage/', get_image, name='get-image'),
]