import os

# Go one directory up (to ai4org root)
os.chdir("..")

# Now run your modules
os.system("python -m data_cleaning_pipeline.main")
os.system("python -m hallucination_reduction.main")
