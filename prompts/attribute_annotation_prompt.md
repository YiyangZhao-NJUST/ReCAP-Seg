### Brain MRI Prompt Template

**Role.**  
You are an experienced radiologist specialized in neuro-oncologic MRI interpretation. Based on the provided brain MRI image and its corresponding lesion mask, generate a structured radiological description of the tumor region. The mask is used only as a region-of-interest cue to localize the lesion and reduce ambiguity from surrounding brain tissues.

**Task.**  
Describe only image-visible characteristics of the masked lesion according to the predefined semantic slots below. Use concise, standardized, and medically appropriate terminology. Do not infer pathological diagnosis, tumor grade, prognosis, treatment response, or any information that is not visually supported by the image.

**Output slots.**

1. **Lesion Distribution:**  
   Describe whether the lesion is solitary or multifocal, and indicate its anatomical distribution if visible. If the lesion extends across multiple lobes or involves deep structures, such as the basal ganglia, corpus callosum, periventricular region, or ventricles, specify this involvement.

2. **Shape:**  
   Describe the overall lesion morphology, such as round, oval, lobulated, regular, or irregular.

3. **Margin Definition:**  
   Assess whether the lesion boundary is well-defined, partially defined, or poorly defined.

4. **Internal Heterogeneity:**  
   Describe whether the lesion appears homogeneous or heterogeneous. If visible, mention necrotic, cystic, hemorrhagic, calcified, or mixed internal components.

5. **Peritumoral Interface:**  
   Describe the relationship between the lesion and adjacent brain tissue, including suspected infiltration, indistinct lesion–tissue interface, or surrounding signal abnormality.

6. **Mass Effect / Edema:**  
   Describe the presence and extent of peritumoral edema or mass effect, including sulcal effacement, ventricular compression, or midline shift, if visible.

7. **Intensity Pattern:**  
   Describe the dominant signal appearance of the lesion on the provided MRI image using only visible imaging characteristics. If sequence-specific interpretation is uncertain, provide a conservative description based on relative signal intensity.

**Output requirements.**

- Return the response strictly in the seven-slot format above.
- For each slot, provide a concise phrase or short sentence.
- If a feature is not visible or cannot be determined from the image, write `not detected`.
- Do not include additional explanations, background information, diagnostic speculation, or recommendations.
- Do not describe regions outside the masked lesion unless they are directly relevant to the lesion–tissue interface, edema, or mass effect.
