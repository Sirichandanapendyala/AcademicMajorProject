import pandas as pd
from django.shortcuts import render
from django.views.generic import TemplateView
import joblib

class HomeView(TemplateView):
    template_name = 'home.html'

def predict(request):
    return render(request, 'index.html')

def results(request):
    if request.method == 'POST':
        # Load the trained model
        model = joblib.load('C:/Users/SIRI CHANDANA/OneDrive/Desktop/major project/placement_app/Random Forest/random_forest_model_gridsearch.joblib')

        # Extract the feature values from the request
        gender = request.POST.get('Gender')
        tenth_percentage = request.POST.get('10th %')
        ssc_board = request.POST.get('SSC_Board')
        internships = request.POST.get('Internships')
        course = request.POST.get('course')
        twelth_stream = request.POST.get('12th_Stream')
        degree_stream = request.POST.get('Degree_Stream')
        degree_percentage = request.POST.get('Degree_%')
        etest_percentage = request.POST.get('E_Test_%')

        # Handle missing values and convert to lowercase
        gender = gender.lower() if gender else 'male'
        ssc_board = ssc_board.lower() if ssc_board else 'others'
        internships = internships.lower() if internships else 'no'
        course = course.lower() if course else 'ai'
        twelth_stream = twelth_stream.lower() if twelth_stream else 'science'  # Assuming default value is 'science'
        degree_stream = degree_stream.lower() if degree_stream else 'sci&tech'
        tenth_percentage = float(tenth_percentage) if tenth_percentage else 0.0
        degree_percentage = float(degree_percentage) if degree_percentage else 0.0
        etest_percentage = float(etest_percentage) if etest_percentage else 0.0

        # Define label encoders for categorical features
        label_encoders = {
            'Gender': {'f': 0, 'm': 1},
            'SSC Board': {'others': 0, 'state': 1},
            'Internships': {'no': 0, 'yes': 1},
            '12th Stream': {'commerce': 0, 'science': 1},
            'Degree Stream': {'commerce': 0, 'sci&tech': 1},
            'course': {'ai': 0, 'blockchain': 1, 'fullstack': 2, 'ml': 3}
        }

        # Convert categorical features to numerical using label encoders
        gender_encoded = label_encoders['Gender'][gender]
        course_encoded = label_encoders['course'][course]
        ssc_board_encoded = label_encoders['SSC Board'][ssc_board]
        internships_encoded = label_encoders['Internships'][internships]
        twelth_stream_encoded = label_encoders['12th Stream'][twelth_stream]
        degree_stream_encoded = label_encoders['Degree Stream'][degree_stream]

        if (gender == 'm' and degree_percentage <= 75.0 and tenth_percentage <= 75.0) or (gender == 'f' and degree_percentage <= 75.0 and tenth_percentage <= 75.0):
            return render(request, 'results.html', {'prediction': 'Not eligible for placements'})
        elif degree_percentage < 80.0 and tenth_percentage <= 75.0 and internships == 'no':
            return render(request, 'results.html', {'prediction': 'Not eligible for placements'})
        elif degree_percentage >= 85.0 and tenth_percentage >= 76.0 and internships == 'no':
            return render(request, 'results.html', {'prediction': 'Eligible for placements'})

        else: 
            prediction = model.predict([[gender_encoded, tenth_percentage, ssc_board_encoded, internships_encoded, twelth_stream_encoded, degree_percentage, degree_stream_encoded, course_encoded, etest_percentage]])
            prediction_placement = 'Eligible for Placements' if prediction[0] == 'Not Placed' else 'Not Placed'
            return render(request, 'results.html', {'prediction': prediction_placement})

    # If request method is not POST or features are invalid, render an empty results page
    return render(request, 'results.html')

def evaluation(request):
    # Define the paths to the images
    image_paths = [
        'images/confusion_matrix (1).png',
        'images/correaltion_matrix_heatmap (1).png',
        'images/feature_importance_bar_graph.png',
        'images/feature_importance_pie_chart.png',
        'images/precision_recall_curve (2).png',
        'images/roc_curve.png'
    ]

    return render(request, 'evaluation.html', {'image_paths': image_paths})
