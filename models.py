from django.db import models

# Create your models here.
class user_reg(models.Model):
    id = models.AutoField(primary_key=True)
    fname = models.CharField(max_length=300)
    lname = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    mobile = models.BigIntegerField()
    uname = models.CharField(max_length=300)
    password = models.CharField(max_length=200)

class ddos_dataset(models.Model):
    id = models.AutoField(primary_key=True)
    ddos_data=models.CharField(max_length=500)
    attack_result=models.CharField(max_length=500)