"""intrusion_detection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from projectuser import views as projectuserviews
from intrusion_detection import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$',projectuserviews.user_index, name="user_index"),
    url(r'^user_login/$',projectuserviews.user_login, name="user_login"),
    url(r'^user_register/$',projectuserviews.user_register, name="user_register"),
    url(r'^user_home/$',projectuserviews.user_home, name="user_home"),
    url(r'^add_data/$',projectuserviews.add_data, name="add_data"),
    url(r'^labeled_data/$',projectuserviews.labeled_data, name="labeled_data"),
    url(r'^unlabeled_data/$',projectuserviews.unlabeled_data, name="unlabeled_data"),
    url(r'^intrusion_analysis/$',projectuserviews.intrusion_analysis, name="intrusion_analysis"),
    url(r'^graphical_analysis/$',projectuserviews.graphical_analysis, name="graphical_analysis"),
    url(r'^random_forest/$',projectuserviews.random_forest, name="random_forest"),
    url(r'^naive_bayes/$',projectuserviews.naive_bayes, name="naive_bayes"),
    url(r'^svm/$',projectuserviews.svm, name="svm"),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
