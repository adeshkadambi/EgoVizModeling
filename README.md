# CDSS Modeling

This repo contains all code for machine learning models pertaining to clinical decision support dashboard developed
for hand function neurorehabilitation in outpatient settings. The primary focus is to classify activities of daily 
living (ADLs) and instrumental activities of daily living (iADLs) that are informative of hand function at home.

#### ADL/iADL categories include:
- Feeding
> Setting up, arranging, and bringing food [or fluid] from the plate or cup to the mouth; sometimes called self-feeding

- Functional Mobility
> Functional mobility Moving from one position or place to another (during performance of everyday activities), 
> such as in-bed mobility, wheelchair mobility, and transfers (e.g., wheelchair, bed, car, shower, tub, toilet, 
> chair, floor) . Includes functional ambulation and transportation of objects.

- Grooming / Health Management
> Obtaining and using supplies; removing body hair (e.g., using razor, tweezers, lotion); 
> applying and removing cosmetics; washing, drying, combing, styling, brushing, and trimming hair; caring for nails 
> (hands and feet); caring for skin, ears, eyes, and nose; applying deodorant; cleaning mouth; brushing and flossing 
> teeth; and removing, cleaning, and reinserting dental orthotics and prosthetics. Developing, managing, and 
> maintaining routines for health and wellness promotion, such as physical fitness, nutrition, decreased health 
> risk behaviors, and medication routines.

- Communication Management
> Sending, receiving, and interpreting information using a variety of systems and equipment, 
> including writing tools, telephones (cell phones or smartphones), keyboards, audiovisual recorders, computers or 
> tablets, communication boards, call lights, emergency systems, Braille writers, telecommunication devices for deaf 
> people, augmentative communication systems, and personal digital assistants.

- Home Establishment and Management
> Obtaining and maintaining personal and household possessions and environment (e.g., home, yard, garden, appliances, 
> vehicles), including maintaining and repairing personal possessions (e .g ., clothing, household items) and knowing 
> how to seek help or whom to contact.

- Meal Preparation and Cleanup
> Planning, preparing, and serving well-balanced, nutritious meals and cleaning up food and utensils after meals.

- Leisure or Other Activities
> Nonobligatory activity that is intrinsically motivated and engaged in during discretionary time, that is, time 
> not committed to obligatory occupations or any of the aforementioned ADLs or iADLs.

*Aota. (2020). Occupational Therapy Practice Framework: Domain & Process. 
American Occupational Therapy Association, Incorporated.*

---
## Models
```
> 100DOH Model - hand object interactions and determination of active objects.

Shan, D., Geng, J., Shu, M., & Fouhey, D. F. (2020). Understanding Human Hands in Contact at Internet Scale. 
In arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2006.06669

> Unidet Model - classification and localization of everyday objects related to functional activity.

Zhou, X., Koltun, V., & Krähenbühl, P. (2021). Simple multi-dataset detection. 
In arXiv [cs.CV]. arXiv. http://arxiv.org/abs/2102.13086
```