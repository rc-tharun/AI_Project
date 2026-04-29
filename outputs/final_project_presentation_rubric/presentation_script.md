# Final Presentation Script

## Slide 1 - Cover / Talk map

"Today I’ll present our final project on ODD plus OOD detection for safe autonomy. The main goal was to study how a perception system should behave when the driving environment shifts, especially under weather conditions like rain, fog, and snow. I’ll first motivate the problem, then summarize the baselines we tried, then present the final method we recommend, followed by the experiments and the conclusion. The key message is that our strongest current story is a hybrid system: ConvNeXt-Tiny for classification and CLIP plus kNN for explicit OOD rejection."

## Slide 2 - Motivation & problem definition

"This problem matters because a model can look accurate in normal conditions and still become dangerously overconfident when the environment changes. In safe autonomy, a wrong prediction with high confidence is much worse than refusing to act. So our framing is not just classification accuracy, but decision-making under domain shift. For each frame, the system should choose one of three actions: trust the prediction, slow down if the evidence is uncertain, or abstain if the frame looks out of distribution."

## Slide 3 - Baseline & related work

"We explored six method families across the notebooks. First, we used ResNet-50 with a small classifier head as a straightforward baseline. Then we tested confidence-based uncertainty methods such as MSP, energy, dropout, and ensemble entropy. We also tried deep ensembles, one-class methods like SVDD, distance-based methods such as Mahalanobis, CLIP plus kNN, and finally a supervised backbone sweep. Our contribution is not inventing a new architecture, but identifying which existing approach gives the most convincing ODD-aware safety behavior."

## Slide 4 - Method

"Based on the evidence, the final method we recommend is a hybrid safety gate with two branches. The first branch uses ConvNeXt-Tiny as the classifier because it gave the best validation accuracy in the backbone sweep. The second branch uses CLIP ViT-B/32 with FAISS kNN because it was the strongest explicit OOD detector in the notebook results. We then combine the calibrated classifier confidence and the kNN distance into a safety gate that outputs trust, slow down, or abstain. One important caveat is that both branches still need to be rerun on one shared benchmark split before making any deployment claim."

## Slide 5 - Experimental setup

"The experiments came from three different notebook tracks, so we present them carefully and honestly. Track A is the curated binary split used for the ResNet baseline and the backbone sweep. Track B uses a clear-weather reference bank and is where the CLIP plus kNN and Mahalanobis detectors were evaluated. Track C is a full-weather binary setup used for the exploratory deep ensemble and SVDD baselines. Because these splits are different, we compare methods directly within a track and only use cross-track comparison as directional guidance."

## Slide 6 - Experiment 1: ResNet baseline and calibration

"The ResNet baseline learned the binary task reasonably well, reaching 0.8415 validation accuracy. Temperature scaling substantially improved calibration, reducing ECE from 0.1037 to 0.0161. However, the best OOD AUROC from the confidence-style methods was only 0.5263, which shows weak separation between in-domain and out-of-domain frames. Even worse, to satisfy a strict false-safe target, the gate had to abstain on about 97.6 percent of validation examples. So the main takeaway is that calibration improved, but confidence-only OOD detection was still not practical."

## Slide 7 - Experiment 2: OOD detector comparison and ablation

"This slide compares the explicit OOD detectors. Deep SVDD performed the weakest among the main baselines shown here. Mahalanobis improved over SVDD, but the CLIP-based kNN methods were clearly better. CLIP ViT-B/32 plus kNN achieved the strongest AUROC at 0.7412, while ViT-L/14 was close and gave slightly better coverage at the same false-safe budget. The k-sweep also shows that performance stabilized around k equals 5, so there was little reason to use a larger value. The conclusion from this slide is that distance-based CLIP embeddings gave the strongest OOD rejection story in our experiments."

## Slide 8 - Experiment 3: backbone comparison

"For the supervised classifier sweep, stronger backbones improved validation accuracy on the curated task. ConvNeXt-Tiny was best at 0.9368, with CLIP ViT-B/16 just behind at 0.9359. ResNet-50 was still solid, while EfficientNet-B3 was the weakest of the four. This tells us that ConvNeXt-Tiny is the strongest classification backbone in our current results. At the same time, this notebook did not report the same reject-option metrics as the CLIP plus kNN notebooks, so it supports the classifier choice rather than the full safety gate by itself."

## Slide 9 - Conclusion & discussion

"To conclude, the notebooks support three main claims. First, ConvNeXt-Tiny is our strongest classifier. Second, CLIP ViT-B/32 plus kNN is our strongest explicit OOD detector. Third, temperature scaling helps calibration, but confidence-only rejection is too conservative to be the final solution. So the final presentation should recommend a hybrid method built from those strongest pieces. The immediate next step is to place every method on one unified evaluation protocol, then measure latency and intervention quality so the gate can be judged as a practical real-time safety component."
