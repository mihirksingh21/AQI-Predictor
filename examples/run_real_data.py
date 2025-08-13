"""
Script to run AQI prediction system with real data
"""

from main_no_tensorflow import AQIPredictionSystemNoTF

def main():
    print("=" * 60)
    print("AQI PREDICTION SYSTEM - REAL DATA RUN")
    print("=" * 60)
    
    system = AQIPredictionSystemNoTF()
    
    # Try different cities that are likely to have data
    cities_to_try = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad"]
    
    for city in cities_to_try:
        print(f"\nTrying to collect data for {city}...")
        
        try:
            # Run the complete pipeline
            results = system.run_complete_pipeline(city=city, days=7)
            
            if results and results['processed_data']:
                print(f"\n✅ SUCCESS! Data collected and models trained for {city}")
                print(f"Total samples: {len(results['processed_data']['merged_data'])}")
                
                # Show model performance
                if results['results']:
                    print("\nModel Performance:")
                    for model_name, (metrics, predictions) in results['results'].items():
                        print(f"{model_name}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}")
                
                return results
            else:
                print(f"⚠ No sufficient data for {city}")
                
        except Exception as e:
            print(f"❌ Error with {city}: {e}")
            continue
    
    print("\n❌ Could not collect sufficient real data for any city.")
    print("Running with demo data instead...")
    
    # Fallback to demo data
    demo_data = system.create_demo_data()
    processed_data = system.preprocessor.preprocess_all(
        demo_data['aqi_data'],
        demo_data['weather_data'],
        demo_data['satellite_data']
    )
    
    if processed_data:
        results = system.train_and_evaluate_models(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            processed_data['X_test'],
            processed_data['y_test']
        )
        
        system.visualizer.create_all_visualizations(
            processed_data['merged_data'],
            results
        )
        
        system.print_results_summary(results, processed_data)
        
        return {
            'data': demo_data,
            'processed_data': processed_data,
            'results': results
        }
    
    return None

if __name__ == "__main__":
    main() 