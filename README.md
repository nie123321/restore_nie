```bash
# Install
pip install -r requirements.txt

# Train
python train.py --data-root xx --output xx

# Resume
python train.py --data-root xx --output xx --resume xx

# Inference (image or folder)
python test.py --checkpoint xx --input xx --output xx

# Evaluate (paired dataset)
python test.py --checkpoint xx --data-root xx --output xx
```
