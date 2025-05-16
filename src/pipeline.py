from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib



class ModelBuilder:
    def __init__(self, numerical, ordinal, onehot):
        print("ModelBuilder initialized")
        
        self.numerical = numerical
        self.ordinal = ordinal
        self.onehot = onehot

    def build_pipeline(self):
        # Preprocessing for numeric features
        num_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])

        # Preprocessing for ordinal features
        ord_transformer = Pipeline([
            ("ordinal", OrdinalEncoder(categories=[['low', 'medium', 'high']]))
        ])

        ohe_transformer = Pipeline([
        ("onehot", OneHotEncoder()) 
    ])

        # Combine all transformations
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, self.numerical),
            ("ord", ord_transformer, self.ordinal),
            ("ohe", ohe_transformer, self.onehot),
        ])

        # Final pipeline: preprocessing + classifier
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=0))
        ])

        return pipeline


    def tune_pipeline(self, X_train, y_train):
        
        pipeline = self.build_pipeline()

        # Define hyperparameter grid
        param_grid = {
            'classifier__max_depth': [3, 5, None], 
            'classifier__max_features': [1.0],
            'classifier__max_samples': [0.7, 1.0],
            'classifier__min_samples_leaf': [1, 2, 3],
            'classifier__min_samples_split': [2, 3, 4],
            'classifier__n_estimators': [300, 500],
        }

        # Define scoring metrics
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        # Grid search
        rf2_pipeline_grid  = GridSearchCV(pipeline, param_grid, scoring=scoring, refit='roc_auc', cv=4)
        rf2_pipeline_grid.fit(X_train, y_train)

        return rf2_pipeline_grid.best_estimator_, rf2_pipeline_grid.best_params_


def save_pipeline(pipeline):
    joblib.dump(pipeline, "rf2_pipeline.pkl")

