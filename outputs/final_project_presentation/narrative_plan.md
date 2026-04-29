Audience: project instructor, technical review audience, and classmates evaluating the final project direction.

Objective: turn the initial proposal deck plus all experiment notebooks into a final presentation that shows what was tried, what actually worked, what did not, and which hybrid direction should be presented as the final recommendation.

Narrative arc:
1. Re-establish the project problem as a safety gate for camera-based autonomy.
2. Explain the dataset and, critically, the fact that the notebooks use different splits.
3. Summarize every major method family from the notebooks.
4. Show that the ResNet confidence-based baseline calibrates well but remains weak at OOD separation.
5. Show that feature-space distance methods, especially CLIP + kNN, are more promising for explicit OOD rejection.
6. Show that ConvNeXt-Tiny is the strongest supervised classifier from the backbone sweep.
7. Recommend a final hybrid pipeline: ConvNeXt-Tiny classifier plus CLIP ViT-B/32 kNN reject gate.
8. Close with the caveat that all final claims should be revalidated on one shared evaluation split.

Slide list:
1. Cover with final synthesized takeaways.
2. Safety gate objective and decision outputs.
3. Dataset definition and three experimental tracks.
4. Methods tried across the notebooks.
5. ResNet baseline findings.
6. Feature-space OOD detector comparison.
7. Supervised backbone comparison.
8. Recommended final pipeline.
9. Supported claims and next steps.

Source plan:
- /Users/crreddy/Documents/AI_Project/ODD-OOD-Detection-for-Safe-Autonomy.pptx
- /Users/crreddy/Documents/AI_Project/ResNet50/AI_RESNET.ipynb
- /Users/crreddy/Documents/AI_Project/Deep_Ensemble/Deep_Ensemble.ipynb
- /Users/crreddy/Documents/AI_Project/SVDD/Deep_Ensemble_(2).ipynb
- /Users/crreddy/Documents/AI_Project/results of vit/Deep_Ensemble_Mahalanobis.ipynb
- /Users/crreddy/Documents/AI_Project/Vit+knn/Deep_Ensemble_(1).ipynb
- /Users/crreddy/Documents/AI_Project/vit_l_14/Deep_Ensemble_(1).ipynb
- /Users/crreddy/Documents/AI_Project/results of vit/VIT_BACKBONE.ipynb
- /Users/crreddy/Documents/AI_Project/ResNet50/method_comparison.png

Visual system:
- Warm paper background with navy primary text.
- Teal, amber, blue, and coral used to distinguish evidence types and method families.
- Title font: Caladea.
- Body font: Lato.
- Layout language: cards, metric panels, banners, and a clean flow diagram.

Image plan:
- Reuse the experiment-generated ResNet method comparison plot as the only raster evidence image.
- Keep all important narrative, labels, numbers, and recommendation text editable in native PowerPoint objects.
- Do not use generated art plates for this deck, because the project plots already provide method-specific visual evidence and the remaining slides are better served by clean technical layouts.

Asset needs:
- method_comparison.png from the ResNet notebook outputs.

Editability plan:
- All titles, subtitles, cards, metrics, callouts, and flow-diagram labels are authored as editable text shapes.
- The experiment plot is embedded as an image, but every critical conclusion is duplicated in editable slide text.
- No chart objects are required because the comparative evidence is communicated through metric cards and one sourced experiment plot.

Honesty guardrail:
- The deck must explicitly distinguish between the curated binary split, the clear-only ID bank split, and the full weather binary split.
- Conclusions should identify the best method within the evidence available, without pretending that all notebook numbers are directly comparable.
