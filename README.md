# embedding_multimodal

To install requirements, run the following command:

pip install -r requirements.txt

To download diffusiondb dataset, run the following command:

python download.py -i 1 -r 5

If you have unzip, you can extract them by running the following command:

unzip -n 'images/*.zip' -d 'extracted'

To run dashboard, run the following command:

python -m streamlit run .\dashboard.py PROJECT-ID