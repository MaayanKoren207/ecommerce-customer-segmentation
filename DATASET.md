# Dataset

## Source

**Ecommerce Consumer Behavior Analysis Data**  
Creator/uploader: Salahuddin Ahmed  
Platform: Kaggle  
License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**

Source page:
https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data

## Local file

`data/ecommerce_consumer_behavior.csv`

The copy included here contains 1,000 rows and 28 columns.

## Attribution

The dataset is redistributed in this repository for reproducibility under its CC BY 4.0 license. The original dataset creator/source should be credited when reusing or redistributing the data.

## Important project fields

### Final clustering inputs
- `Time_of_Purchase` → transformed into **RECENCY**
- `Frequency_of_Purchase` → **FREQUENCY**
- `Purchase_Amount` → cleaned into numeric **MONETARY**

### Held out from clustering / post-hoc interpretation
- `Customer_Satisfaction`
- `Purchase_Intent`
- demographics
- channel/device/payment/shipping fields
- loyalty and discount behavior

The modeling decision was deliberate: the final segmentation is intended to describe customer purchase value/behavior rather than reproduce demographic or operational categories.
