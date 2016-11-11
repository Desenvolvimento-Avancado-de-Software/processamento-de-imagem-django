from django.conf.urls import url
from django.contrib import admin
from face_detector.views import detect

urlpatterns = [
    # Examples:

    url(r'^face_detection/detect/$', detect, name='detect'),

    # url(r'^$', 'cv_api.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', admin.site.urls),
]
