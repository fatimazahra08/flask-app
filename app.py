from flask import Flask, request, render_template  # type: ignore
import pandas as pd  # type: ignore
import joblib  # type: ignore
import matplotlib # type: ignore
matplotlib.use('Agg')  # ← AJOUTE CECI ICI

import matplotlib.pyplot as plt # type: ignore
import io
import base64

import base64
import numpy as np  # type: ignore
app = Flask(__name__)

# Load all models
rf_model = joblib.load('model/randomrf.pkl')                    # Random Forest
lr_model = joblib.load('model/linear_regression_pipeline.pkl') # Linear Regression pipeline
pls_model = joblib.load('model/pls_pipeline.pkl')              # PLS Regression pipeline
gb_model = joblib.load('model/gb_pipeline.pkl')                # Gradient Boosting / Lasso pipeline
ridge_model = joblib.load('model/ridge_pipeline.pkl')          # Ridge pipeline

def create_model_wins_plot(actual, predictions):
    import matplotlib.pyplot as plt # type: ignore
    import io
    import base64

    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)

    # Calculer l'erreur absolue
    errors = pred_df.sub(actual, axis=0).abs()

    # Trouver le modèle qui a la meilleure prédiction pour chaque voiture
    best_models = errors.idxmin(axis=1)

    # Compter les victoires pour chaque modèle
    win_counts = best_models.value_counts().reindex(pred_df.columns, fill_value=0)

    # Couleurs cohérentes
    color_map = {
        'Random Forest': 'tab:blue',
        'Linear Regression': 'tab:orange',
        'PLS': 'tab:green',
        'Gradient Boosting': 'tab:red',
        'Ridge': 'tab:purple'
    }
    colors = [color_map.get(model, 'gray') for model in win_counts.index]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(win_counts.index, win_counts.values, color=colors)
    plt.xlabel("Modèle")
    plt.ylabel("Nombre de meilleures prédictions")
    plt.title("Nombre de fois où chaque modèle a été le plus proche du prix réel")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convertir l’image en base64 pour HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64


def create_comparison_plot(actual, predictions):
  
    from matplotlib.patches import Patch # type: ignore

    plt.figure(figsize=(14, 6))
    num_cars = len(actual)
    models = list(predictions.keys())

    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(predictions)

    # Find best model for each car
    errors = pred_df.sub(actual, axis=0).abs()
    best_models = errors.idxmin(axis=1)

    # Get best predictions (fixed version)
    best_preds = pred_df.to_numpy()[np.arange(len(pred_df)), pred_df.columns.get_indexer(best_models)]

    # Assign colors to models
    color_map = {
        'Random Forest': 'tab:blue',
        'Linear Regression': 'tab:orange',
        'PLS': 'tab:green',
        'Gradient Boosting': 'tab:red',
        'Ridge': 'tab:purple'
    }

    # Plot best prediction bars
    indices = np.arange(num_cars)
    bar_colors = [color_map[model] for model in best_models]
    plt.bar(indices, best_preds, color=bar_colors)

    # Plot actual price line
    plt.plot(indices, actual, color='black', linestyle='--', marker='o', label='Prix réel')

    # Create a custom legend
    legend_elements = [Patch(facecolor=color, label=model) for model, color in color_map.items()]
    legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='--', marker='o', label='Prix réel'))
    plt.legend(handles=legend_elements)

    plt.xlabel('Index voiture')
    plt.ylabel('Prix')
    plt.title('Meilleure prédiction par voiture comparée au prix réel')
    plt.tight_layout()

    # Convert to base64 for HTML embedding
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file, encoding='ISO-8859-1', sep=',')
            original_df = df.copy()

            # Clean and preprocess numeric columns
            for col in ['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())

            # Keep selling price
            actual_prices = df['selling_price'].tolist() if 'selling_price' in df.columns else None

            # One-hot encode for models that need it
            df_encoded = pd.get_dummies(df.drop(columns=['selling_price'], errors='ignore'))

            predictions = {}

            # --- Random Forest ---
            try:
                predictions['Random Forest'] = rf_model.predict(df).tolist()
                original_df['RF_Prediction'] = predictions['Random Forest']
            except Exception as e:
                original_df['RF_Prediction'] = f'Error: {e}'

            # --- Linear Regression ---
            try:
                lr_input = df[['mileage(km/ltr/kg)']]
                predictions['Linear Regression'] = lr_model.predict(lr_input).tolist()
                original_df['LinearRegression'] = predictions['Linear Regression']
            except Exception as e:
                original_df['LinearRegression'] = f'Error: {e}'

            # --- PLS Regression ---
            try:
                pls_features = pls_model.feature_names_in_
                for col in pls_features:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_pls = df_encoded[pls_features]
                predictions['PLS'] = pls_model.predict(df_pls).flatten().tolist()
                original_df['PLS'] = predictions['PLS']
            except Exception as e:
                original_df['PLS'] = f'Error: {e}'

            # --- Gradient Boosting ---
            try:
                gb_features = gb_model.feature_names_in_
                for col in gb_features:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_gb = df_encoded[gb_features]
                predictions['Gradient Boosting'] = gb_model.predict(df_gb).tolist()
                original_df['GB'] = predictions['Gradient Boosting']
            except Exception as e:
                original_df['GB'] = f'Error: {e}'

            # --- Ridge Regression ---
            try:
                ridge_features = ridge_model.feature_names_in_
                missing_cols = [col for col in ridge_features if col not in df_encoded.columns]
                if missing_cols:
                    zeros_df = pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)
                    df_encoded = pd.concat([df_encoded, zeros_df], axis=1)
                df_ridge = df_encoded[ridge_features]
                predictions['Ridge'] = ridge_model.predict(df_ridge).tolist()
                original_df['Ridge'] = predictions['Ridge']
            except Exception as e:
                original_df['Ridge'] = f'Error: {e}'

            # Generate chart
            chart_img = None
            if actual_prices and isinstance(actual_prices, list):
                chart_img = create_comparison_plot(actual_prices, predictions)

            # Convert to HTML table
            result_table = original_df.to_html(
                classes='table table-bordered table-striped',
                index=False,
                float_format='{:,.2f}'.format
            )
            # Créer les deux visualisations
            comparison_plot = create_comparison_plot(actual_prices, predictions)
            wins_plot = create_model_wins_plot(actual_prices, predictions)

            # Envoyer les deux images au template
            return render_template(
                'results.html',
                table=result_table,
                plot_image=comparison_plot,
                wins_image=wins_plot
            )


            

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
