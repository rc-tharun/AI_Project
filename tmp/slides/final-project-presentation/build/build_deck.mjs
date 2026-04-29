const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;

const DECK_ID = "final-project-presentation-visual";
const OUT_DIR = "/Users/crreddy/Documents/AI_Project/outputs/final_project_presentation_visual";
const SCRATCH_DIR = path.resolve("/Users/crreddy/Documents/AI_Project/tmp/slides/final-project-presentation-visual");
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const VERIFICATION_DIR = path.join(SCRATCH_DIR, "verification");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");
const MAX_RENDER_VERIFY_LOOPS = 3;

const COLORS = {
  navy: "#10212B",
  navySoft: "#173848",
  teal: "#0F766E",
  tealSoft: "#DCEFEA",
  amber: "#D97706",
  amberSoft: "#FFF2DD",
  coral: "#C94E38",
  coralSoft: "#FBE4DF",
  blue: "#1E6091",
  blueSoft: "#E4F0F8",
  paper: "#F6F4EE",
  paperStrong: "#FFFDF8",
  ink: "#13212C",
  graphite: "#44515A",
  muted: "#6F7A81",
  white: "#FFFFFF",
  black: "#000000",
  transparent: "#00000000",
  whiteAlpha: "#FFFFFFD9",
  whiteAlphaStrong: "#FFFFFFEF",
  tealAlpha: "#0F766E1A",
  navyAlpha: "#10212B14",
  amberAlpha: "#D977061A",
};

const TITLE_FACE = "Caladea";
const BODY_FACE = "Lato";
const MONO_FACE = "Aptos Mono";

const ASSETS = {
  resnetMethodComparison: "/Users/crreddy/Documents/AI_Project/ResNet50/method_comparison.png",
  fogRoad: "/Users/crreddy/Documents/AI_Project/outputs/final_project_presentation_rubric/assets/fog-road.png",
  snowRoad: "/Users/crreddy/Documents/AI_Project/outputs/final_project_presentation_rubric/assets/snow-road.png",
  glareRoad: "/Users/crreddy/Documents/AI_Project/outputs/final_project_presentation_rubric/assets/glare-road.png",
};

const SOURCES = {
  initial_deck:
    "/Users/crreddy/Documents/AI_Project/ODD-OOD-Detection-for-Safe-Autonomy.pptx (project framing, goal, and original slide arc)",
  resnet_nb:
    "/Users/crreddy/Documents/AI_Project/ResNet50/AI_RESNET.ipynb (ResNet-50 baseline, uncertainty scoring, temperature scaling, saved plots)",
  deep_ensemble_nb:
    "/Users/crreddy/Documents/AI_Project/Deep_Ensemble/Deep_Ensemble.ipynb (5-head ensemble on ResNet features, 91.38% validation accuracy on full weather binary split)",
  svdd_nb:
    "/Users/crreddy/Documents/AI_Project/SVDD/Deep_Ensemble_(2).ipynb (Deep SVDD baseline: AUROC 0.5841, coverage 0.0625 at 5% false-safe)",
  mahal_nb:
    "/Users/crreddy/Documents/AI_Project/results of vit/Deep_Ensemble_Mahalanobis.ipynb (Mahalanobis and centroid-distance OOD detection on ResNet features)",
  clip_b32_nb:
    "/Users/crreddy/Documents/AI_Project/Vit+knn/Deep_Ensemble_(1).ipynb (CLIP ViT-B/32 + FAISS kNN: AUROC 0.7412, coverage 0.1904 at 5% false-safe)",
  clip_l14_nb:
    "/Users/crreddy/Documents/AI_Project/vit_l_14/Deep_Ensemble_(1).ipynb (CLIP ViT-L/14 + kNN sweep; best k=5, AUROC 0.7326)",
  backbone_nb:
    "/Users/crreddy/Documents/AI_Project/results of vit/VIT_BACKBONE.ipynb (supervised backbone sweep: ConvNeXt-Tiny best val accuracy 0.9368)",
};

const inspectRecords = [];

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  if (!bytes.byteLength) {
    throw new Error(`Image file is empty: ${imagePath}`);
  }
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function normalizeImageConfig(config) {
  if (!config.path) {
    return config;
  }
  const { path: imagePath, ...rest } = config;
  return {
    ...rest,
    blob: await readImageBlob(imagePath),
  };
}

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(SCRATCH_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(VERIFICATION_DIR, { recursive: true });
}

function lineConfig(fill = COLORS.transparent, width = 0) {
  return { style: "solid", fill, width };
}

function normalizeText(text) {
  if (Array.isArray(text)) {
    return text.map((item) => String(item ?? "")).join("\n");
  }
  return String(text ?? "");
}

function textLineCount(text) {
  const value = normalizeText(text);
  if (!value.trim()) {
    return 0;
  }
  return Math.max(1, value.split(/\n/).length);
}

function requiredTextHeight(text, fontSize, lineHeight = 1.18, minHeight = 8) {
  const lines = textLineCount(text);
  if (lines === 0) {
    return minHeight;
  }
  return Math.max(minHeight, lines * fontSize * lineHeight);
}

function assertTextFits(text, boxHeight, fontSize, role = "text") {
  const required = requiredTextHeight(text, fontSize);
  const tolerance = Math.max(2, fontSize * 0.08);
  if (normalizeText(text).trim() && boxHeight + tolerance < required) {
    throw new Error(
      `${role} text box is too short: height=${boxHeight.toFixed(1)}, required>=${required.toFixed(1)}, ` +
        `lines=${textLineCount(text)}, fontSize=${fontSize}, text=${JSON.stringify(normalizeText(text).slice(0, 120))}`,
    );
  }
}

function wrapText(text, widthChars) {
  const paragraphs = normalizeText(text).split("\n");
  return paragraphs
    .map((paragraph) => {
      const words = paragraph.split(/\s+/).filter(Boolean);
      const lines = [];
      let current = "";
      for (const word of words) {
        const next = current ? `${current} ${word}` : word;
        if (next.length > widthChars && current) {
          lines.push(current);
          current = word;
        } else {
          current = next;
        }
      }
      if (current) {
        lines.push(current);
      }
      return lines.join("\n");
    })
    .join("\n");
}

function formatLines(lines, widthChars = 38) {
  return lines.map((line) => wrapText(line, widthChars)).join("\n");
}

function recordShape(slideNo, shape, role, shapeType, x, y, w, h) {
  if (!slideNo) return;
  inspectRecords.push({
    kind: "shape",
    slide: slideNo,
    id: shape?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    shapeType,
    bbox: [x, y, w, h],
  });
}

function recordText(slideNo, shape, role, text, x, y, w, h) {
  const value = normalizeText(text);
  inspectRecords.push({
    kind: "textbox",
    slide: slideNo,
    id: shape?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    text: value,
    textPreview: value.replace(/\n/g, " | ").slice(0, 220),
    textChars: value.length,
    textLines: textLineCount(value),
    bbox: [x, y, w, h],
  });
}

function recordImage(slideNo, image, role, imagePath, x, y, w, h) {
  inspectRecords.push({
    kind: "image",
    slide: slideNo,
    id: image?.id || `slide-${slideNo}-${role}-${inspectRecords.length + 1}`,
    role,
    path: imagePath,
    bbox: [x, y, w, h],
  });
}

function addShape(slide, geometry, x, y, w, h, fill = COLORS.transparent, line = COLORS.transparent, lineWidth = 0, meta = {}) {
  const shape = slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: lineConfig(line, lineWidth),
  });
  recordShape(meta.slideNo, shape, meta.role || geometry, geometry, x, y, w, h);
  return shape;
}

function applyTextStyle(box, text, size, color, bold, face, align, valign, autoFit) {
  box.text = text;
  box.text.fontSize = size;
  box.text.color = color;
  box.text.bold = Boolean(bold);
  box.text.alignment = align;
  box.text.verticalAlignment = valign;
  box.text.typeface = face;
  box.text.insets = { left: 0, right: 0, top: 0, bottom: 0 };
  if (autoFit) {
    box.text.autoFit = autoFit;
  }
}

function addText(
  slide,
  slideNo,
  text,
  x,
  y,
  w,
  h,
  {
    size = 20,
    color = COLORS.ink,
    bold = false,
    face = BODY_FACE,
    align = "left",
    valign = "top",
    fill = COLORS.transparent,
    line = COLORS.transparent,
    lineWidth = 0,
    autoFit = null,
    checkFit = true,
    role = "text",
  } = {},
) {
  if (checkFit) {
    assertTextFits(text, h, size, role);
  }
  const box = addShape(slide, "rect", x, y, w, h, fill, line, lineWidth, { slideNo, role });
  applyTextStyle(box, text, size, color, bold, face, align, valign, autoFit);
  recordText(slideNo, box, role, text, x, y, w, h);
  return box;
}

async function addImage(slide, slideNo, config, position, role, sourcePath = null) {
  const image = slide.images.add(await normalizeImageConfig(config));
  image.position = position;
  recordImage(
    slideNo,
    image,
    role,
    sourcePath || config.path || config.uri || "inline-data-url",
    position.left,
    position.top,
    position.width,
    position.height,
  );
  return image;
}

function addTag(slide, slideNo, text, x, y, w, accent = COLORS.teal, dark = false) {
  addShape(slide, "roundRect", x, y, w, 26, accent, accent, 0, { slideNo, role: `tag: ${text}` });
  addText(slide, slideNo, text.toUpperCase(), x + 10, y + 5, w - 20, 16, {
    size: 11,
    color: dark ? COLORS.ink : COLORS.white,
    bold: true,
    face: MONO_FACE,
    checkFit: false,
    role: "tag text",
  });
}

function decorateLightSlide(slide, slideNo, accent = COLORS.teal) {
  slide.background.fill = COLORS.paper;
  addShape(slide, "ellipse", 1010, -70, 250, 250, COLORS.navyAlpha, COLORS.transparent, 0, {
    slideNo,
    role: "decorative ellipse",
  });
  addShape(slide, "ellipse", -70, 540, 200, 200, COLORS.amberAlpha, COLORS.transparent, 0, {
    slideNo,
    role: "decorative ellipse",
  });
  addShape(slide, "rect", 64, 58, 1152, 2, COLORS.navy, COLORS.transparent, 0, { slideNo, role: "header rule" });
  addShape(slide, "rect", 64, 58, 76, 5, accent, COLORS.transparent, 0, { slideNo, role: "header accent" });
}

function addHeader(slide, slideNo, kicker, idx, total, accent = COLORS.teal) {
  addText(slide, slideNo, kicker.toUpperCase(), 64, 28, 500, 20, {
    size: 12,
    color: accent,
    bold: true,
    face: MONO_FACE,
    checkFit: false,
    role: "header",
  });
  addText(slide, slideNo, `${String(idx).padStart(2, "0")} / ${String(total).padStart(2, "0")}`, 1088, 28, 128, 20, {
    size: 12,
    color: accent,
    bold: true,
    face: MONO_FACE,
    align: "right",
    checkFit: false,
    role: "header",
  });
}

function addTitleBlock(slide, slideNo, title, subtitle, x = 64, y = 82, w = 920, dark = false) {
  addText(slide, slideNo, title, x, y, w, 116, {
    size: 38,
    color: dark ? COLORS.white : COLORS.navy,
    bold: true,
    face: TITLE_FACE,
    role: "title",
  });
  if (subtitle) {
    addText(slide, slideNo, subtitle, x + 2, y + 104, Math.min(w, 880), 62, {
      size: 18,
      color: dark ? COLORS.white : COLORS.graphite,
      face: BODY_FACE,
      autoFit: "shrinkText",
      checkFit: false,
      role: "subtitle",
    });
  }
}

function addMetricCard(slide, slideNo, x, y, w, h, metric, label, note = null, accent = COLORS.teal) {
  addShape(slide, "roundRect", x, y, w, h, COLORS.whiteAlphaStrong, COLORS.navy, 1.1, { slideNo, role: `metric card: ${label}` });
  addShape(slide, "rect", x, y, w, 7, accent, COLORS.transparent, 0, { slideNo, role: `metric accent: ${label}` });
  addText(slide, slideNo, metric, x + 20, y + 22, w - 40, 42, {
    size: 31,
    color: COLORS.navy,
    bold: true,
    face: TITLE_FACE,
    role: "metric value",
  });
  addText(slide, slideNo, label, x + 20, y + 68, w - 40, 34, {
    size: 15,
    color: COLORS.graphite,
    face: BODY_FACE,
    role: "metric label",
  });
  if (note) {
    addText(slide, slideNo, note, x + 20, y + h - 30, w - 40, 20, {
      size: 10,
      color: COLORS.muted,
      face: BODY_FACE,
      checkFit: false,
      role: "metric note",
    });
  }
}

function addInfoCard(
  slide,
  slideNo,
  x,
  y,
  w,
  h,
  {
    tag = null,
    tagWidth = 120,
    title,
    subtitle = null,
    lines = [],
    footer = null,
    accent = COLORS.teal,
    fill = COLORS.whiteAlphaStrong,
    bodyWidthChars = 40,
    role = "info card",
  },
) {
  addShape(slide, "roundRect", x, y, w, h, fill, COLORS.navy, 1.1, { slideNo, role });
  addShape(slide, "rect", x, y, 8, h, accent, COLORS.transparent, 0, { slideNo, role: `${role} accent` });
  let titleY = y + 22;
  if (tag) {
    addTag(slide, slideNo, tag, x + 20, y + 18, tagWidth, accent);
    titleY = y + 54;
  }
  addText(slide, slideNo, title, x + 22, titleY, w - 44, 30, {
    size: 23,
    color: COLORS.navy,
    bold: true,
    face: TITLE_FACE,
    role: `${role} title`,
  });
  let bodyY = titleY + 36;
  if (subtitle) {
    addText(slide, slideNo, subtitle, x + 22, bodyY, w - 44, 34, {
      size: 13,
      color: accent,
      bold: true,
      face: MONO_FACE,
      role: `${role} subtitle`,
    });
    bodyY += 42;
  }
  const footerSpace = footer ? 48 : 18;
  const body = formatLines(lines, bodyWidthChars);
  addText(slide, slideNo, body, x + 22, bodyY, w - 44, h - (bodyY - y) - footerSpace, {
    size: 14,
    color: COLORS.ink,
    face: BODY_FACE,
    autoFit: "shrinkText",
    checkFit: false,
    role: `${role} body`,
  });
  if (footer) {
    addShape(slide, "rect", x + 20, y + h - 42, w - 40, 1.5, COLORS.navyAlpha, COLORS.transparent, 0, {
      slideNo,
      role: `${role} footer rule`,
    });
    addText(slide, slideNo, footer, x + 22, y + h - 31, w - 44, 18, {
      size: 10,
      color: COLORS.muted,
      face: BODY_FACE,
      checkFit: false,
      role: `${role} footer`,
    });
  }
}

function addDecisionStep(slide, slideNo, x, y, w, h, title, body, accent, fill) {
  addShape(slide, "roundRect", x, y, w, h, fill, accent, 1.2, { slideNo, role: `decision step: ${title}` });
  addText(slide, slideNo, title, x + 22, y + 18, w - 44, 28, {
    size: 20,
    color: accent,
    bold: true,
    face: TITLE_FACE,
    role: "decision title",
  });
  addText(slide, slideNo, wrapText(body, 48), x + 22, y + 48, w - 44, h - 60, {
    size: 14,
    color: COLORS.ink,
    face: BODY_FACE,
    autoFit: "shrinkText",
    checkFit: false,
    role: "decision body",
  });
}

function addImagePanel(slide, slideNo, imagePath, x, y, w, h, title = null) {
  addShape(slide, "roundRect", x, y, w, h, COLORS.whiteAlphaStrong, COLORS.navy, 1.1, { slideNo, role: `image panel: ${title || imagePath}` });
  if (title) {
    addText(slide, slideNo, title, x + 20, y + 16, w - 40, 22, {
      size: 12,
      color: COLORS.teal,
      bold: true,
      face: MONO_FACE,
      checkFit: false,
      role: "image caption",
    });
  }
  return addImage(
    slide,
    slideNo,
    { path: imagePath, fit: "contain", alt: title || "Project plot" },
    { left: x + 14, top: y + 42, width: w - 28, height: h - 56 },
    "project plot",
    imagePath,
  );
}

async function addPhotoCard(slide, slideNo, imagePath, x, y, w, h, label, accent = COLORS.teal) {
  addShape(slide, "roundRect", x, y, w, h, COLORS.whiteAlphaStrong, COLORS.navy, 1.1, {
    slideNo,
    role: `photo card: ${label}`,
  });
  await addImage(
    slide,
    slideNo,
    { path: imagePath, fit: "cover", alt: label },
    { left: x + 10, top: y + 10, width: w - 20, height: h - 44 },
    `photo plate: ${label}`,
    imagePath,
  );
  addShape(slide, "roundRect", x + 14, y + h - 28, 78, 18, accent, accent, 0, {
    slideNo,
    role: `photo label badge: ${label}`,
  });
  addText(slide, slideNo, label.toUpperCase(), x + 24, y + h - 25, 58, 10, {
    size: 9,
    color: COLORS.white,
    bold: true,
    face: MONO_FACE,
    checkFit: false,
    role: `photo label: ${label}`,
  });
}

function addBanner(slide, slideNo, text, x, y, w, h, accent = COLORS.navy, fill = COLORS.tealSoft) {
  addShape(slide, "roundRect", x, y, w, h, fill, accent, 1.1, { slideNo, role: "banner" });
  addText(slide, slideNo, text, x + 18, y + 10, w - 36, h - 20, {
    size: 16,
    color: accent,
    bold: true,
    face: BODY_FACE,
    valign: "middle",
    autoFit: "shrinkText",
    checkFit: false,
    role: "banner text",
  });
}

function addAgendaRow(slide, slideNo, x, y, w, time, label, accent = COLORS.teal, fill = COLORS.whiteAlphaStrong) {
  addShape(slide, "roundRect", x, y, w, 54, fill, COLORS.navy, 1.1, { slideNo, role: `agenda row: ${label}` });
  addShape(slide, "roundRect", x + 10, y + 10, 82, 34, accent, accent, 0, { slideNo, role: `agenda time badge: ${label}` });
  addText(slide, slideNo, time, x + 22, y + 18, 58, 16, {
    size: 12,
    color: COLORS.white,
    bold: true,
    face: MONO_FACE,
    checkFit: false,
    role: "agenda time",
  });
  addText(slide, slideNo, label, x + 110, y + 14, w - 130, 24, {
    size: 17,
    color: COLORS.navy,
    bold: true,
    face: BODY_FACE,
    checkFit: false,
    role: "agenda label",
  });
}

function addComparisonTable(slide, slideNo, x, y, colWidths, headers, rows, accent = COLORS.teal, role = "comparison table") {
  const totalWidth = colWidths.reduce((sum, width) => sum + width, 0);
  const headerH = 36;
  const rowH = 44;
  const totalHeight = headerH + rows.length * rowH;
  addShape(slide, "roundRect", x, y, totalWidth, totalHeight, COLORS.whiteAlphaStrong, COLORS.navy, 1.1, { slideNo, role });

  let currentX = x;
  headers.forEach((header, idx) => {
    addShape(slide, "rect", currentX, y, colWidths[idx], headerH, idx === 0 ? accent : COLORS.navy, COLORS.transparent, 0, {
      slideNo,
      role: `${role} header`,
    });
    addText(slide, slideNo, header, currentX + 10, y + 10, colWidths[idx] - 20, 16, {
      size: 12,
      color: COLORS.white,
      bold: true,
      face: MONO_FACE,
      checkFit: false,
      role: `${role} header text`,
    });
    currentX += colWidths[idx];
  });

  rows.forEach((row, rowIdx) => {
    const rowY = y + headerH + rowIdx * rowH;
    const rowFill = rowIdx % 2 === 0 ? COLORS.whiteAlphaStrong : COLORS.paperStrong;
    let cellX = x;
    row.forEach((cell, cellIdx) => {
      addShape(slide, "rect", cellX, rowY, colWidths[cellIdx], rowH, rowFill, COLORS.navyAlpha, 0.6, {
        slideNo,
        role: `${role} cell`,
      });
      addText(slide, slideNo, String(cell), cellX + 10, rowY + 9, colWidths[cellIdx] - 20, rowH - 16, {
        size: 13,
        color: COLORS.ink,
        face: BODY_FACE,
        autoFit: "shrinkText",
        checkFit: false,
        role: `${role} cell text`,
      });
      cellX += colWidths[cellIdx];
    });
  });
}

function addArrow(slide, slideNo, x, y, w, h, fill = COLORS.teal) {
  addShape(slide, "rightArrow", x, y, w, h, fill, fill, 0, { slideNo, role: "flow arrow" });
}

function addFlowBox(slide, slideNo, x, y, w, h, title, body, accent = COLORS.teal) {
  addShape(slide, "roundRect", x, y, w, h, COLORS.whiteAlphaStrong, COLORS.navy, 1.1, { slideNo, role: `flow box: ${title}` });
  addShape(slide, "rect", x, y, w, 8, accent, COLORS.transparent, 0, { slideNo, role: `flow accent: ${title}` });
  addText(slide, slideNo, title, x + 18, y + 18, w - 36, 28, {
    size: 19,
    color: COLORS.navy,
    bold: true,
    face: TITLE_FACE,
    role: "flow title",
  });
  addText(slide, slideNo, wrapText(body, Math.max(16, Math.floor(w / 12))), x + 18, y + 54, w - 36, h - 70, {
    size: 14,
    color: COLORS.ink,
    face: BODY_FACE,
    autoFit: "shrinkText",
    checkFit: false,
    role: "flow body",
  });
}

function addNotes(slide, body, sourceKeys) {
  const sourceLines = (sourceKeys || []).map((key) => `- ${SOURCES[key] || key}`).join("\n");
  slide.speakerNotes.setText(`${body}\n\n[Sources]\n${sourceLines}`);
}

async function slideCover(presentation) {
  const slideNo = 1;
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.paper;

  addShape(slide, "rect", 0, 0, 760, H, COLORS.navy, COLORS.transparent, 0, { slideNo, role: "cover dark panel" });
  addShape(slide, "ellipse", -80, -120, 320, 320, COLORS.teal, COLORS.transparent, 0, { slideNo, role: "cover ellipse" });
  addShape(slide, "ellipse", 560, 520, 230, 230, COLORS.amber, COLORS.transparent, 0, { slideNo, role: "cover ellipse" });
  addShape(slide, "rect", 760, 0, 520, H, COLORS.paperStrong, COLORS.transparent, 0, { slideNo, role: "cover light panel" });

  addText(slide, slideNo, "FINAL PRESENTATION", 72, 72, 260, 18, {
    size: 12,
    color: "#A7E3DA",
    bold: true,
    face: MONO_FACE,
    checkFit: false,
    role: "cover kicker",
  });
  addText(slide, slideNo, "ODD + OOD Detection for Safe Autonomy", 72, 114, 620, 170, {
    size: 47,
    color: COLORS.white,
    bold: true,
    face: TITLE_FACE,
    role: "cover title",
  });
  addText(
    slide,
    slideNo,
    "Final project presentation refactored around the grading rubric: motivation, related work, method, experiments, and conclusion.",
    72,
    298,
    560,
    78,
    {
      size: 19,
      color: COLORS.white,
      face: BODY_FACE,
      role: "cover subtitle",
    },
  );

  addInfoCard(slide, slideNo, 72, 402, 540, 224, {
    tag: "Project aim",
    tagWidth: 100,
    title: "Target contribution",
    lines: [
      "- Build an ODD-aware safety gate for camera-based autonomy under weather shift.",
      "- Show which methods actually worked in the notebooks, not just what was proposed.",
      "- Present one technically honest final recommendation with clear next steps.",
    ],
    accent: COLORS.teal,
    fill: COLORS.whiteAlphaStrong,
    role: "cover objective",
    bodyWidthChars: 52,
  });

  addText(slide, slideNo, "Talk map and pacing", 806, 78, 360, 28, {
    size: 24,
    color: COLORS.navy,
    bold: true,
    face: TITLE_FACE,
    role: "cover right title",
  });
  addAgendaRow(slide, slideNo, 804, 126, 392, "0:45", "Motivation and problem definition", COLORS.teal);
  addAgendaRow(slide, slideNo, 804, 192, 392, "1:00", "Baselines and related work", COLORS.blue);
  addAgendaRow(slide, slideNo, 804, 258, 392, "1:00", "Proposed final method", COLORS.amber);
  addAgendaRow(slide, slideNo, 804, 324, 392, "3:00", "Experiments and evidence", COLORS.coral);
  addAgendaRow(slide, slideNo, 804, 390, 392, "1:00", "Conclusion and discussion", COLORS.teal);
  addBanner(slide, slideNo, "Target total: about 7 minutes, leaving buffer for questions and transitions.", 804, 462, 392, 64, COLORS.navy, COLORS.blueSoft);
  addBanner(
    slide,
    slideNo,
    "Key message for the audience: the best current story is a hybrid system with ConvNeXt-Tiny for classification and CLIP + kNN for explicit OOD rejection.",
    804,
    546,
    392,
    88,
    COLORS.navy,
    COLORS.whiteAlphaStrong,
  );

  addNotes(
    slide,
    "Cover slide with the talk map to address pacing and time management directly. Spend under one minute here, then move into the motivation slide.",
    ["initial_deck", "resnet_nb", "clip_b32_nb", "backbone_nb", "svdd_nb", "mahal_nb", "clip_l14_nb", "deep_ensemble_nb"],
  );
}

async function slideObjective(presentation) {
  const slideNo = 2;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.teal);
  addHeader(slide, slideNo, "Motivation & problem definition", slideNo, 9, COLORS.teal);
  addTitleBlock(
    slide,
    slideNo,
    "Why this problem matters",
  );

  addInfoCard(slide, slideNo, 64, 196, 520, 320, {
    tag: "Problem",
    tagWidth: 82,
    title: "Non-trivial safety challenge",
    lines: [
      "- The same road scene can become visually unfamiliar under weather shift.",
      "- False-safe predictions are worse than abstaining, because the system acts with unjustified confidence.",
      "- The gate must work per frame, so the solution has to be accurate and lightweight.",
      "- We treat the task as ODD-aware decision making, not only image classification.",
    ],
    footer: "Original framing adapted from the initial presentation deck.",
    accent: COLORS.teal,
    role: "motivation card",
    bodyWidthChars: 47,
  });

  await addPhotoCard(slide, slideNo, ASSETS.fogRoad, 620, 196, 180, 150, "Fog", COLORS.teal);
  await addPhotoCard(slide, slideNo, ASSETS.snowRoad, 816, 196, 180, 150, "Snow", COLORS.blue);
  await addPhotoCard(slide, slideNo, ASSETS.glareRoad, 1012, 196, 180, 150, "Glare", COLORS.amber);

  addDecisionStep(slide, slideNo, 620, 360, 572, 82, "TRUST", "Classifier is confident and the OOD score stays below threshold.", COLORS.teal, COLORS.tealSoft);
  addDecisionStep(slide, slideNo, 620, 454, 572, 82, "SLOW DOWN", "Prediction is usable, but the margin is narrow and the OOD score is near threshold.", COLORS.amber, COLORS.amberSoft);
  addDecisionStep(slide, slideNo, 620, 548, 572, 82, "ABSTAIN", "Reject the frame when the representation looks out-of-domain or uncertainty spikes.", COLORS.coral, COLORS.coralSoft);

  addBanner(
    slide,
    slideNo,
    "Problem definition: given one front-camera frame, decide TRUST, SLOW DOWN, or ABSTAIN using both classification confidence and out-of-domain evidence.",
    64,
    640,
    1124,
    40,
    COLORS.navy,
    COLORS.whiteAlphaStrong,
  );

  addNotes(
    slide,
    "Motivation slide aligned to the rubric. Emphasize why the problem is challenging, safety-critical, and worth solving.",
    ["initial_deck"],
  );
}

async function slideDataset(presentation) {
  const slideNo = 5;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.amber);
  addHeader(slide, slideNo, "Experiments", slideNo, 9, COLORS.amber);
  addTitleBlock(
    slide,
    slideNo,
    "Experimental setup and evaluation protocol",
  );

  addBanner(
    slide,
    slideNo,
    "Reporting rule for this talk: compare methods within a track directly, and use cross-track comparisons only as directional guidance.",
    64,
    172,
    1152,
    52,
    COLORS.coral,
    COLORS.coralSoft,
  );

  addInfoCard(slide, slideNo, 64, 244, 354, 276, {
    tag: "Track A",
    tagWidth: 82,
    title: "Curated binary split",
    subtitle: "ResNet baseline plus backbone sweep",
    lines: [
      "- Train: 16,000 frames (10,000 ID / 6,000 OOD).",
      "- Val: 3,988 frames (2,500 ID / 1,488 OOD).",
      "- OOD buckets: rain and fog/snow.",
      "- Bucketed corpus in the backbone notebook totals 70,587 images.",
    ],
    footer: "Used by AI_RESNET.ipynb and VIT_BACKBONE.ipynb",
    accent: COLORS.teal,
    role: "dataset track a",
    bodyWidthChars: 30,
  });

  addInfoCard(slide, slideNo, 463, 244, 354, 276, {
    tag: "Track B",
    tagWidth: 82,
    title: "ID-only reference bank",
    subtitle: "CLIP + kNN and Mahalanobis detectors",
    lines: [
      "- Train ID bank: 37,344 clear-weather frames.",
      "- Validation: 5,346 ID and 1,520 OOD frames.",
      "- OOD weather in val: rainy, snowy, foggy.",
      "- Goal: measure distance from a clean in-domain feature bank.",
    ],
    footer: "Used by Vit+knn, vit_l_14, and Deep_Ensemble_Mahalanobis notebooks",
    accent: COLORS.amber,
    role: "dataset track b",
    bodyWidthChars: 30,
  });

  addInfoCard(slide, slideNo, 862, 244, 354, 276, {
    tag: "Track C",
    tagWidth: 82,
    title: "Full weather binary map",
    subtitle: "Exploratory ensemble and SVDD baselines",
    lines: [
      "- Train: 61,744 frames.",
      "- Validation: 8,843 frames.",
      "- ID mapping: clear, partly cloudy, overcast.",
      "- OOD mapping: rainy, snowy, foggy.",
    ],
    footer: "Used by Deep_Ensemble.ipynb and SVDD/Deep_Ensemble_(2).ipynb",
    accent: COLORS.coral,
    role: "dataset track c",
    bodyWidthChars: 30,
  });

  await addPhotoCard(slide, slideNo, ASSETS.fogRoad, 64, 540, 354, 98, "Fog", COLORS.teal);
  await addPhotoCard(slide, slideNo, ASSETS.snowRoad, 463, 540, 354, 98, "Snow", COLORS.blue);
  await addPhotoCard(slide, slideNo, ASSETS.glareRoad, 862, 540, 354, 98, "Glare", COLORS.amber);

  addBanner(
    slide,
    slideNo,
    "Metrics used throughout the talk: AUROC, AUPR, FPR@95TPR, calibration error, false-safe rate, and coverage under a false-safe budget.",
    64,
    650,
    1152,
    34,
    COLORS.navy,
    COLORS.whiteAlphaStrong,
  );

  addNotes(
    slide,
    "Experimental setup slide. This is where to explain the three evaluation tracks and why split differences matter.",
    ["resnet_nb", "backbone_nb", "clip_b32_nb", "clip_l14_nb", "mahal_nb", "deep_ensemble_nb", "svdd_nb"],
  );
}

async function slideMethods(presentation) {
  const slideNo = 3;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.blue);
  addHeader(slide, slideNo, "Baseline & related work", slideNo, 9, COLORS.blue);
  addTitleBlock(
    slide,
    slideNo,
    "Baselines and related work that position our contribution",
  );

  const cardW = 352;
  const cardH = 188;
  const xs = [64, 464, 864];
  const yTop = 220;
  const yBottom = 432;

  addInfoCard(slide, slideNo, xs[0], yTop, cardW, cardH, {
    tag: "ResNet",
    tagWidth: 84,
    title: "ResNet-50 + MLP",
    lines: [
      "- Frozen ImageNet features with a binary head.",
      "- Main baseline for calibration and uncertainty scoring.",
    ],
    accent: COLORS.teal,
    role: "method resnet",
    bodyWidthChars: 31,
  });

  addInfoCard(slide, slideNo, xs[1], yTop, cardW, cardH, {
    tag: "Uncertainty",
    tagWidth: 112,
    title: "Confidence scores",
    lines: [
      "- MSP, energy, MC dropout, and ensemble entropy.",
      "- Best AUROC was still only 0.5263.",
    ],
    accent: COLORS.amber,
    role: "method uncertainty",
    bodyWidthChars: 31,
  });

  addInfoCard(slide, slideNo, xs[2], yTop, cardW, cardH, {
    tag: "Ensemble",
    tagWidth: 92,
    title: "Deep ensemble",
    lines: [
      "- Five MLP heads on cached ResNet features.",
      "- 91.38% validation accuracy on the full-weather split.",
    ],
    accent: COLORS.coral,
    role: "method deep ensemble",
    bodyWidthChars: 31,
  });

  addInfoCard(slide, slideNo, xs[0], yBottom, cardW, cardH, {
    tag: "One-class",
    tagWidth: 98,
    title: "SVDD + Mahalanobis",
    lines: [
      "- One-class and distance baselines around the ID feature cloud.",
      "- Mahalanobis beat SVDD, but both trailed CLIP + kNN.",
    ],
    accent: COLORS.blue,
    role: "method one class",
    bodyWidthChars: 31,
  });

  addInfoCard(slide, slideNo, xs[1], yBottom, cardW, cardH, {
    tag: "CLIP",
    tagWidth: 70,
    title: "CLIP + kNN",
    lines: [
      "- Distance-to-ID in CLIP embedding space.",
      "- Strongest OOD detector family in the notebooks.",
    ],
    accent: COLORS.teal,
    role: "method clip",
    bodyWidthChars: 31,
  });

  addInfoCard(slide, slideNo, xs[2], yBottom, cardW, cardH, {
    tag: "Backbones",
    tagWidth: 108,
    title: "Backbone sweep",
    lines: [
      "- ResNet50, EfficientNet-B3, ConvNeXt-Tiny, and CLIP ViT-B/16.",
      "- ConvNeXt-Tiny was best at 0.9368 validation accuracy.",
    ],
    accent: COLORS.amber,
    role: "method backbone sweep",
    bodyWidthChars: 31,
  });

  addBanner(
    slide,
    slideNo,
    "Positioning statement: our contribution is not inventing a new backbone, but identifying which existing families support an ODD-aware safety gate most convincingly.",
    64,
    644,
    1152,
    46,
    COLORS.navy,
    COLORS.tealSoft,
  );

  addNotes(
    slide,
    "Related-work slide aligned to the rubric. Present these six families as the baseline space and explain how the final contribution is positioned among them.",
    ["resnet_nb", "deep_ensemble_nb", "svdd_nb", "mahal_nb", "clip_b32_nb", "clip_l14_nb", "backbone_nb"],
  );
}

async function slideResNetFindings(presentation) {
  const slideNo = 6;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.teal);
  addHeader(slide, slideNo, "Experiments", slideNo, 9, COLORS.teal);
  addTitleBlock(
    slide,
    slideNo,
    "Experiment 1: ResNet baseline and calibration",
  );

  addMetricCard(slide, slideNo, 64, 222, 238, 112, "0.8415", "Best val accuracy", "Curated binary split", COLORS.teal);
  addMetricCard(slide, slideNo, 320, 222, 238, 112, "0.5263", "Best OOD AUROC", "Ensemble entropy", COLORS.amber);
  addMetricCard(slide, slideNo, 64, 350, 238, 112, "0.0161", "ECE after scaling", "From 0.1037 before T-scaling", COLORS.blue);
  addMetricCard(slide, slideNo, 320, 350, 238, 112, "97.6%", "Abstain rate at 5% FS", "Energy gate on curated split", COLORS.coral);

  addBanner(
    slide,
    slideNo,
    "Calibration improved a lot, but OOD separation stayed weak and the gate had to abstain on almost every validation frame to hit the false-safe target.",
    64,
    512,
    494,
    126,
    COLORS.navy,
    COLORS.blueSoft,
  );

  await addImagePanel(slide, slideNo, ASSETS.resnetMethodComparison, 592, 220, 624, 432, "ResNet uncertainty method comparison");

  addNotes(
    slide,
    "ResNet baseline results slide. This uses the saved comparison plot from the ResNet notebook plus the extracted summary metrics from the final results cell.",
    ["resnet_nb"],
  );
}

async function slideFeatureDetectors(presentation) {
  const slideNo = 7;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.amber);
  addHeader(slide, slideNo, "Experiments", slideNo, 9, COLORS.amber);
  addTitleBlock(
    slide,
    slideNo,
    "Experiment 2: OOD detector comparison and ablation",
  );

  addBanner(
    slide,
    slideNo,
    "Most rows below are on the clear-only ID bank split. Deep SVDD is included as an exploratory baseline from the full-weather split and should be interpreted separately.",
    64,
    184,
    1152,
    48,
    COLORS.navy,
    COLORS.amberSoft,
  );

  addComparisonTable(
    slide,
    slideNo,
    64,
    254,
    [220, 120, 90, 90, 130],
    ["Detector", "Split", "AUROC", "AUPR", "Cov@5%FS"],
    [
      ["Deep SVDD", "Full weather", "0.5841", "0.2454", "0.0625"],
      ["Mahalanobis", "Clear-only", "0.6525", "0.3391", "0.1381"],
      ["CLIP ViT-B/32 + kNN", "Clear-only", "0.7412", "0.4366", "0.1904"],
      ["CLIP ViT-L/14 + k=5", "Clear-only", "0.7326", "0.4175", "0.1936"],
    ],
    COLORS.teal,
    "ood comparison table",
  );

  addComparisonTable(
    slide,
    slideNo,
    810,
    254,
    [70, 120, 150],
    ["k", "AUROC", "Cov@5%FS"],
    [
      ["1", "0.7283", "0.1739"],
      ["5", "0.7326", "0.1936"],
      ["10", "0.7325", "0.1936"],
      ["20", "0.7317", "0.1936"],
      ["50", "0.7300", "0.1904"],
    ],
    COLORS.amber,
    "k sweep table",
  );

  addInfoCard(slide, slideNo, 810, 514, 406, 136, {
    title: "Interpretation",
    lines: [
      "- CLIP-based kNN is the strongest detector family in the notebook set.",
      "- The ViT-L/14 ablation shows only marginal gains beyond k=5.",
    ],
    accent: COLORS.navySoft,
    role: "ood interpretation",
    bodyWidthChars: 40,
  });

  addBanner(
    slide,
    slideNo,
    "Experiment takeaway: CLIP + kNN is the most convincing OOD gate in the current notebooks, and the k-sweep suggests k=5 is a reasonable operating point.",
    64,
    654,
    1152,
    36,
    COLORS.navy,
    COLORS.whiteAlphaStrong,
  );

  addNotes(
    slide,
    "Feature-space comparison slide. SVDD is included as an exploratory baseline and explicitly labeled as a different split; Mahalanobis and both CLIP cards share the clear-only ID bank split.",
    ["svdd_nb", "mahal_nb", "clip_b32_nb", "clip_l14_nb"],
  );
}

async function slideBackboneSweep(presentation) {
  const slideNo = 8;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.blue);
  addHeader(slide, slideNo, "Experiments", slideNo, 9, COLORS.blue);
  addTitleBlock(
    slide,
    slideNo,
    "Experiment 3: backbone comparison and final model selection",
  );

  addBanner(
    slide,
    slideNo,
    "Bucketed corpus in the backbone notebook: 70,587 total images across in_odd, rain, and fog_snow buckets.",
    64,
    176,
    1152,
    48,
    COLORS.navy,
    COLORS.blueSoft,
  );

  addMetricCard(slide, slideNo, 96, 264, 512, 164, "0.9368", "ConvNeXt-Tiny", "Best val accuracy in backbone sweep", COLORS.teal);
  addMetricCard(slide, slideNo, 672, 264, 512, 164, "0.9359", "CLIP ViT-B/16", "Very close second", COLORS.amber);
  addMetricCard(slide, slideNo, 96, 454, 512, 164, "0.9253", "ResNet-50", "Strong baseline, but not the top classifier", COLORS.blue);
  addMetricCard(slide, slideNo, 672, 454, 512, 164, "0.8975", "EfficientNet-B3", "Lowest validation accuracy in this sweep", COLORS.coral);

  addBanner(
    slide,
    slideNo,
    "Takeaway: ConvNeXt-Tiny is the strongest classifier, but this notebook did not report the same reject-option metrics as the CLIP + kNN notebooks.",
    64,
    640,
    1152,
    44,
    COLORS.navy,
    COLORS.whiteAlphaStrong,
  );

  addNotes(
    slide,
    "Backbone sweep slide driven by the final training output in VIT_BACKBONE.ipynb. The conclusion is classification-focused rather than reject-option focused.",
    ["backbone_nb"],
  );
}

async function slideRecommendation(presentation) {
  const slideNo = 4;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.teal);
  addHeader(slide, slideNo, "Method", slideNo, 9, COLORS.teal);
  addTitleBlock(
    slide,
    slideNo,
    "Proposed final method for the presentation",
  );

  addTag(slide, slideNo, "Classifier path", 64, 198, 118, COLORS.blue);
  addTag(slide, slideNo, "OOD path", 64, 396, 94, COLORS.amber);

  addFlowBox(slide, slideNo, 64, 226, 164, 118, "Frame", "Camera frame enters both paths.", COLORS.blue);
  addArrow(slide, slideNo, 246, 271, 52, 22, COLORS.blue);
  addFlowBox(slide, slideNo, 314, 216, 240, 138, "ConvNeXt-Tiny", "Best classifier in the sweep. Outputs an ID/OOD score.", COLORS.blue);
  addArrow(slide, slideNo, 574, 271, 52, 22, COLORS.blue);
  addFlowBox(slide, slideNo, 642, 216, 214, 138, "Temperature scaling", "Calibrate confidence before final gate logic.", COLORS.blue);

  addFlowBox(slide, slideNo, 64, 424, 164, 118, "Frame", "Same frame is embedded for a distance check.", COLORS.amber);
  addArrow(slide, slideNo, 246, 469, 52, 22, COLORS.amber);
  addFlowBox(slide, slideNo, 314, 414, 240, 138, "CLIP ViT-B/32", "Best explicit OOD detector when paired with kNN.", COLORS.amber);
  addArrow(slide, slideNo, 574, 469, 52, 22, COLORS.amber);
  addFlowBox(slide, slideNo, 642, 414, 214, 138, "FAISS kNN", "Compare to the 37,344-image clear-weather ID bank.", COLORS.amber);

  addFlowBox(slide, slideNo, 902, 292, 314, 208, "Safety gate logic", "TRUST below tau. SLOW DOWN near tau. ABSTAIN when distance spikes or confidence collapses.", COLORS.teal);

  addMetricCard(slide, slideNo, 64, 590, 238, 88, "0.9368", "Classifier val acc", null, COLORS.blue);
  addMetricCard(slide, slideNo, 324, 590, 238, 88, "0.7412", "OOD AUROC", null, COLORS.teal);
  addMetricCard(slide, slideNo, 584, 590, 238, 88, "0.1020", "Current kNN tau", null, COLORS.amber);
  addBanner(
    slide,
    slideNo,
    "One required follow-up: rerun both branches on one shared train/val/test protocol before any deployment claim.",
    844,
    584,
    372,
    94,
    COLORS.navy,
    COLORS.whiteAlphaStrong,
  );

  addNotes(
    slide,
    "Method slide aligned to the rubric. Present this as the technically justified final method, motivated by the evidence shown in later experiments.",
    ["backbone_nb", "clip_b32_nb", "resnet_nb"],
  );
}

async function slideConclusion(presentation) {
  const slideNo = 9;
  const slide = presentation.slides.add();
  decorateLightSlide(slide, slideNo, COLORS.coral);
  addHeader(slide, slideNo, "Conclusion & discussion", slideNo, 9, COLORS.coral);
  addTitleBlock(
    slide,
    slideNo,
    "Conclusion and discussion",
  );

  addInfoCard(slide, slideNo, 64, 218, 546, 356, {
    tag: "Supported claims",
    tagWidth: 130,
    title: "What the experiments already support",
    lines: [
      "- ConvNeXt-Tiny was the strongest supervised backbone at 0.9368 validation accuracy.",
      "- CLIP ViT-B/32 + kNN was the strongest explicit OOD gate at 0.7412 AUROC.",
      "- Temperature scaling improved ResNet calibration from 0.1037 ECE to 0.0161.",
      "- Confidence-only OOD scores were too conservative to make a practical gate.",
    ],
    accent: COLORS.teal,
    role: "supported claims",
    bodyWidthChars: 41,
  });

  addInfoCard(slide, slideNo, 670, 218, 546, 356, {
    tag: "Next steps",
    tagWidth: 104,
    title: "What should happen next",
    lines: [
      "- Rebuild one shared benchmark split and rerun every method on it.",
      "- Add night, construction, and corruption shifts for a broader OOD story.",
      "- Report latency and memory so the gate can be judged as a real-time component.",
      "- Integrate the thresholded gate into the simulator and measure intervention quality.",
    ],
    accent: COLORS.amber,
    role: "next steps",
    bodyWidthChars: 41,
  });

  addBanner(
    slide,
    slideNo,
    "Final recommendation for the deck: present ConvNeXt-Tiny as the best classifier, CLIP ViT-B/32 + kNN as the best reject-option OOD detector, and a unified evaluation protocol as the immediate next milestone.",
    64,
    612,
    1152,
    58,
    COLORS.navy,
    COLORS.coralSoft,
  );

  addNotes(
    slide,
    "Closing slide. It translates the notebook review into exactly what the team can claim now and what still needs to be done for a rigorous final result.",
    ["resnet_nb", "deep_ensemble_nb", "svdd_nb", "mahal_nb", "clip_b32_nb", "clip_l14_nb", "backbone_nb"],
  );
}

async function createDeck() {
  await ensureDirs();
  for (const [key, assetPath] of Object.entries(ASSETS)) {
    if (!(await pathExists(assetPath))) {
      throw new Error(`Missing required asset ${key}: ${assetPath}`);
    }
  }

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  await slideCover(presentation);
  await slideObjective(presentation);
  await slideMethods(presentation);
  await slideRecommendation(presentation);
  await slideDataset(presentation);
  await slideResNetFindings(presentation);
  await slideFeatureDetectors(presentation);
  await slideBackboneSweep(presentation);
  await slideConclusion(presentation);

  return presentation;
}

async function saveBlobToFile(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function writeInspectArtifact(presentation) {
  inspectRecords.unshift({
    kind: "deck",
    id: DECK_ID,
    slideCount: presentation.slides.count,
    slideSize: { width: W, height: H },
  });
  presentation.slides.items.forEach((slide, index) => {
    inspectRecords.splice(index + 1, 0, {
      kind: "slide",
      slide: index + 1,
      id: slide?.id || `slide-${index + 1}`,
    });
  });
  const lines = inspectRecords.map((record) => JSON.stringify(record)).join("\n") + "\n";
  await fs.writeFile(INSPECT_PATH, lines, "utf8");
}

async function currentRenderLoopCount() {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  if (!(await pathExists(logPath))) return 0;
  const previous = await fs.readFile(logPath, "utf8");
  return previous.split(/\r?\n/).filter((line) => line.trim()).length;
}

async function nextRenderLoopNumber() {
  return (await currentRenderLoopCount()) + 1;
}

async function appendRenderVerifyLoop(presentation, previewPaths, pptxPath) {
  const logPath = path.join(VERIFICATION_DIR, "render_verify_loops.ndjson");
  const priorCount = await currentRenderLoopCount();
  const record = {
    kind: "render_verify_loop",
    deckId: DECK_ID,
    loop: priorCount + 1,
    maxLoops: MAX_RENDER_VERIFY_LOOPS,
    capReached: priorCount + 1 >= MAX_RENDER_VERIFY_LOOPS,
    timestamp: new Date().toISOString(),
    slideCount: presentation.slides.count,
    previewCount: previewPaths.length,
    previewDir: PREVIEW_DIR,
    inspectPath: INSPECT_PATH,
    pptxPath,
  };
  await fs.appendFile(logPath, JSON.stringify(record) + "\n", "utf8");
  return record;
}

async function verifyAndExport(presentation) {
  await ensureDirs();
  const nextLoop = await nextRenderLoopNumber();
  if (nextLoop > MAX_RENDER_VERIFY_LOOPS) {
    throw new Error(
      `Render/verify/fix loop cap reached: ${MAX_RENDER_VERIFY_LOOPS} total renders are allowed. ` +
        "Do not rerender; note any remaining visual issues in the final response.",
    );
  }

  await writeInspectArtifact(presentation);
  const previewPaths = [];
  for (let idx = 0; idx < presentation.slides.items.length; idx += 1) {
    const slide = presentation.slides.items[idx];
    const preview = await presentation.export({ slide, format: "png", scale: 1 });
    const previewPath = path.join(PREVIEW_DIR, `slide-${String(idx + 1).padStart(2, "0")}.png`);
    await saveBlobToFile(preview, previewPath);
    previewPaths.push(previewPath);
  }

  const pptxBlob = await PresentationFile.exportPptx(presentation);
  const pptxPath = path.join(OUT_DIR, "output.pptx");
  await pptxBlob.save(pptxPath);
  const loopRecord = await appendRenderVerifyLoop(presentation, previewPaths, pptxPath);
  return { pptxPath, loopRecord };
}

const presentation = await createDeck();
const result = await verifyAndExport(presentation);
console.log(result.pptxPath);
