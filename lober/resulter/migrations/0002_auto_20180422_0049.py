# Generated by Django 2.0.4 on 2018-04-22 00:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resulter', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='svm',
            name='kernel',
            field=models.CharField(max_length=32),
        ),
    ]
