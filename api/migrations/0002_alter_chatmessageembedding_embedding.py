# Generated by Django 5.2 on 2025-04-28 12:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='chatmessageembedding',
            name='embedding',
            field=models.JSONField(),
        ),
    ]
