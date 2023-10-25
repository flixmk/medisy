from utils.pipeline import Pipeline

def main():
    # Your code here
    pipeline = Pipeline(save_path="/home/flix/Documents/medisy_testing_ground", classes_to_generate=["CNV", "DME", "DRUSEN", "NORMAL"])
    pipeline.run(num_runs_per_class=10, num_images_per_prompt=4)

if __name__ == '__main__':
    main()
