# Generated by Django 2.0.4 on 2018-04-22 22:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resulter', '0003_auto_20180422_2107'),
    ]

    operations = [
        migrations.AddField(
            model_name='result',
            name='data_length',
            field=models.IntegerField(default=0),
        ),
    ]
