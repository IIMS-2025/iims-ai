# ===== SALES PREDICTION ML IMPLEMENTATION GUIDE =====
# Comprehensive approach for training ML models on synthetic sales data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ===== 1. DATA PREPARATION AND FEATURE ENGINEERING =====
def prepare_features(df):
    """
    Comprehensive feature engineering for sales prediction
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # ===== HANDLE CATEGORICAL COLUMNS UPFRONT =====
    # Convert any categorical columns to string to avoid setitem errors
    categorical_cols = df.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        if col not in ['date']:  # Don't convert date columns
            df[col] = df[col].astype(str)
    
    # ===== TEMPORAL FEATURES =====
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Day of week encoding (handle categorical columns properly)
    dow_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    # Convert to string first if it's categorical, then map
    if df['day_of_week'].dtype.name == 'category':
        df['day_of_week'] = df['day_of_week'].astype(str)
    
    df['dow_encoded'] = df['day_of_week'].map(dow_mapping)
    
    # Weekend flag (handle categorical properly)
    weekend_days = ['Saturday', 'Sunday']
    if df['day_of_week'].dtype.name == 'category':
        df['is_weekend'] = df['day_of_week'].astype(str).isin(weekend_days).astype(int)
    else:
        df['is_weekend'] = df['day_of_week'].isin(weekend_days).astype(int)
    
    # ===== LAG FEATURES =====
    # Sort by date for proper lag calculation
    df = df.sort_values(['tenant_id', 'location_id', 'product_id', 'date'])
    
    # Create lag features
    for days in [1, 3, 7, 14]:
        df[f'sales_lag_{days}'] = df.groupby(['tenant_id', 'location_id', 'product_id'])['target_sales_qty'].shift(days)
    
    # Rolling statistics
    for window in [3, 7, 14]:
        df[f'sales_rolling_mean_{window}'] = df.groupby(['tenant_id', 'location_id', 'product_id'])['target_sales_qty'].rolling(window=window, min_periods=1).mean().values
        df[f'sales_rolling_std_{window}'] = df.groupby(['tenant_id', 'location_id', 'product_id'])['target_sales_qty'].rolling(window=window, min_periods=1).std().values
    
    # ===== PRICE FEATURES =====
    # Handle price categorization properly
    df['price_category'] = pd.cut(df['current_price'], bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    df['price_category'] = df['price_category'].astype(str)  # Convert to string to avoid categorical issues
    
    # Price relative to category average
    if df['product_category'].dtype.name == 'category':
        category_price_mean = df.groupby(df['product_category'].astype(str))['current_price'].transform('mean')
    else:
        category_price_mean = df.groupby('product_category')['current_price'].transform('mean')
    df['price_vs_category_avg'] = df['current_price'] / category_price_mean
    
    # ===== INTERACTION FEATURES =====
    # Convert categorical columns to string for concatenation
    tenant_str = df['tenant_id'].astype(str)
    location_str = df['location_id'].astype(str)
    category_str = df['product_category'].astype(str) if df['product_category'].dtype.name == 'category' else df['product_category']
    
    df['tenant_location'] = tenant_str + '_' + location_str
    df['category_location'] = category_str + '_' + location_str
    df['price_times_holiday'] = df['current_price'] * df['is_holiday'].astype(float)
    
    # ===== CATEGORICAL ENCODING =====
    categorical_features = ['tenant_id', 'location_id', 'product_category', 'price_category', 
                           'tenant_location', 'category_location']
    
    label_encoders = {}
    for feature in categorical_features:
        if feature in df.columns:
            le = LabelEncoder()
            # Convert to string first to handle any categorical dtype issues
            feature_values = df[feature].astype(str)
            df[f'{feature}_encoded'] = le.fit_transform(feature_values)
            label_encoders[feature] = le
    
    # Fill NaN values for lag features
    df = df.fillna(0)
    
    return df, label_encoders

# ===== 2. MODEL IMPLEMENTATIONS =====
class SalesPredictor:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.performance = {}
        
    def prepare_data(self, df, target_col='target_sales_qty', test_size=0.2):
        """Prepare train/test split with proper time-based splitting"""
        
        # Sort by date for proper time series split
        df = df.sort_values('date')
        
        # Features to exclude from training
        exclude_features = ['date', 'day_of_week', 'product_category', 'tenant_location', 
                           'category_location', 'price_category', target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_features]
        
        # Time-based split (last 20% of dates for testing)
        split_date = df['date'].quantile(0.8)
        
        train_data = df[df['date'] <= split_date]
        test_data = df[df['date'] > split_date]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Train date range: {train_data['date'].min()} to {train_data['date'].max()}")
        print(f"Test date range: {test_data['date'].min()} to {test_data['date'].max()}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model with hyperparameter tuning"""
        
        # XGBoost parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42
        }
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Predictions
        y_pred = model.predict(dtest)
        
        # Store results
        self.models['XGBoost'] = model
        self.feature_importance['XGBoost'] = model.get_score(importance_type='weight')
        self.performance['XGBoost'] = self.evaluate_model(y_test, y_pred, 'XGBoost')
        
        return model, y_pred
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        
        model = lgb.LGBMRegressor(
            objective='regression',
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_test)
        
        self.models['LightGBM'] = model
        self.feature_importance['LightGBM'] = dict(zip(X_train.columns, model.feature_importances_))
        self.performance['LightGBM'] = self.evaluate_model(y_test, y_pred, 'LightGBM')
        
        return model, y_pred
    
    def train_catboost(self, X_train, y_train, X_test, y_test):
        """Train CatBoost model"""
        
        # Identify categorical features
        cat_features = [i for i, col in enumerate(X_train.columns) if 'encoded' in col]
        
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            cat_features=cat_features,
            random_seed=42,
            verbose=False
        )
        
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        
        y_pred = model.predict(X_test)
        
        self.models['CatBoost'] = model
        self.feature_importance['CatBoost'] = dict(zip(X_train.columns, model.feature_importances_))
        self.performance['CatBoost'] = self.evaluate_model(y_test, y_pred, 'CatBoost')
        
        return model, y_pred
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        self.models['RandomForest'] = model
        self.feature_importance['RandomForest'] = dict(zip(X_train.columns, model.feature_importances_))
        self.performance['RandomForest'] = self.evaluate_model(y_test, y_pred, 'RandomForest')
        
        return model, y_pred
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Comprehensive model evaluation"""
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"\n{model_name} Performance:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²:   {r2:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
    
    def compare_models(self):
        """Compare all trained models"""
        
        if not self.performance:
            print("No models trained yet!")
            return
            
        comparison = pd.DataFrame(self.performance).T
        comparison = comparison.sort_values('R2', ascending=False)
        
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        print(comparison.round(4))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = comparison[metric].values
            models = comparison.index
            
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(f'{metric} Comparison')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return comparison
    
    def plot_feature_importance(self, model_name='XGBoost', top_n=15):
        """Plot feature importance for specified model"""
        
        if model_name not in self.feature_importance:
            print(f"Model {model_name} not found!")
            return
            
        importance = self.feature_importance[model_name]
        
        # Convert to DataFrame and sort
        imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
        imp_df = imp_df.sort_values('Importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(imp_df['Feature'], imp_df['Importance'])
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()

# ===== 3. MAIN EXECUTION PIPELINE =====
def run_ml_pipeline(df):
    """Complete ML pipeline execution"""
    
    print("="*60)
    print("SALES PREDICTION ML PIPELINE")
    print("="*60)
    
    # 1. Feature Engineering
    print("\n1. Feature Engineering...")
    df_features, label_encoders = prepare_features(df)
    print(f"   Created {df_features.shape[1]} features from {df.shape[1]} original columns")
    
    # 2. Data Preparation
    print("\n2. Preparing train/test split...")
    predictor = SalesPredictor()
    X_train, X_test, y_train, y_test, feature_cols = predictor.prepare_data(df_features)
    
    # 3. Train Multiple Models
    print("\n3. Training models...")
    
    print("\n   Training XGBoost...")
    predictor.train_xgboost(X_train, y_train, X_test, y_test)
    
    print("\n   Training LightGBM...")
    predictor.train_lightgbm(X_train, y_train, X_test, y_test)
    
    print("\n   Training CatBoost...")
    predictor.train_catboost(X_train, y_train, X_test, y_test)
    
    print("\n   Training Random Forest...")
    predictor.train_random_forest(X_train, y_train, X_test, y_test)
    
    # 4. Model Comparison
    print("\n4. Comparing models...")
    comparison = predictor.compare_models()
    
    # 5. Feature Importance Analysis
    print("\n5. Feature importance analysis...")
    best_model = comparison.index[0]  # Best performing model
    predictor.plot_feature_importance(best_model)
    
    return predictor, comparison, df_features

# ===== 4. ADVANCED ANALYSIS FUNCTIONS =====
def analyze_predictions(y_true, y_pred, model_name):
    """Detailed prediction analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Prediction Analysis', fontsize=16)
    
    # Actual vs Predicted
    axes[0,0].scatter(y_true, y_pred, alpha=0.6)
    axes[0,0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Actual Sales')
    axes[0,0].set_ylabel('Predicted Sales')
    axes[0,0].set_title('Actual vs Predicted')
    
    # Residuals
    residuals = y_true - y_pred
    axes[0,1].scatter(y_pred, residuals, alpha=0.6)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Predicted Sales')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residual Plot')
    
    # Residual distribution
    axes[1,0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Residual Distribution')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/datasets/enhanced_synthetic_sales_data.csv')
    
    # Run complete pipeline
    predictor, comparison, df_features = run_ml_pipeline(df)
    
    # Analyze best model predictions
    best_model_name = comparison.index[0]