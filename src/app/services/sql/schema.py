DATABASE_SCHEMA = """
crop:
  id: Unique crop ID
  crop_name: English crop name (e.g., Boro Paddy, Aman Rice, Wheat, Potato)
  crop_bangla_name: Bangla crop name (e.g., বোরো ধান, আমন ধান, গম, আলু)
  crop_category_id: Reference to crop_categories.id

crop_category:
  id: Unique category ID
  name: Crop category name (e.g., Fruits, Vegetables, Spices)

variety:
  id: Unique variety ID
  crop_id: Parent crop reference
  variety_name: Complete variety name (e.g., ব্রি ধান ২৮, বারি আলু ৮, নাগরাজ)
  rating: Variety rating
  price: Variety price

pesticide:
  id: Unique record ID
  crop_id: Associated crop reference
  disease_name: Pest or disease name
  disease_type: Pest or disease classification (e.g., insect, fungus, bacteria)

herbicide:
  id: Unique record ID
  crop_id: Associated crop reference
  weed_name: Weed name (e.g., শ্যামা ঘাস, মুথা ঘাস, দূর্বা ঘাস)
  company_id: Company reference

chemical:
  id: Unique chemical ID
  pesticide_id: Associated pesticide record
  trade_name: Commercial chemical product name (e.g., রিপকর্ড, টিডো, একতারা)
  generic_name: Generic chemical name
  company_id: Manufacturer company reference

company:
  id: Unique company ID
  name: Company name (e.g., এসিআই ক্রপ কেয়ার লিমিটেড, পদ্মা ওয়েল কোম্পানী)

Relationships:
  varieties.crop_id -> crops.id
  crops.crop_category_id -> crop_categories.id
  pesticides.crop_id -> crops.id
  herbicides.crop_id -> crops.id
  chemicals.pesticide_id -> pesticides.id
  chemicals.company_id -> company.id
"""