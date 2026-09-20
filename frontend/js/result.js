// ─────────────────────────────────────────────────────────
//  DermaAI — Result Page Logic
// ─────────────────────────────────────────────────────────

// Disease info database (supplement backend data)
const DISEASE_INFO = {
  "Acne": {
    emoji: "🧴",
    severity: "Mild to Moderate",
    icd10: "L70",
    description: "Acne vulgaris is a common skin condition that occurs when hair follicles become plugged with oil and dead skin cells. It causes whiteheads, blackheads, or pimples and most often affects adolescents, though it can affect people of all ages.",
    signs: ["Whiteheads (closed clogged pores)", "Blackheads (open clogged pores)", "Small red, tender bumps (papules)", "Pimples (pustules)", "Large, solid, painful lumps (nodules)"],
    care: [
      "Wash the affected area gently with a mild cleanser twice daily",
      "Avoid touching, squeezing, or popping pimples",
      "Use non-comedogenic moisturizers and sunscreen",
      "Remove makeup before sleeping",
      "Stay hydrated and maintain a balanced diet",
    ],
    when_to_consult: "If acne is severe, painful, causing scarring, or not improving with over-the-counter products after 2–3 months.",
    color: "#f59e0b"
  },
  "Eczema": {
    emoji: "🩹",
    severity: "Mild to Severe",
    icd10: "L20",
    description: "Eczema (atopic dermatitis) is a chronic inflammatory skin condition that causes itchy, red, and rough patches. It often flares periodically and may be accompanied by asthma or hay fever.",
    signs: ["Dry, sensitive skin", "Intense itching, especially at night", "Red to brownish-gray patches", "Small, raised bumps that may weep fluid", "Thickened, cracked, or scaly skin"],
    care: [
      "Moisturize skin at least twice a day with fragrance-free cream",
      "Avoid known triggers (soaps, detergents, dust, pet dander)",
      "Take lukewarm baths and pat skin gently dry",
      "Wear soft, breathable fabrics (cotton)",
      "Keep fingernails short to prevent skin damage from scratching",
    ],
    when_to_consult: "If symptoms are severe, sleep-disrupting, covering large areas, or showing signs of infection (yellow crusting, fever).",
    color: "#3b82f6"
  },
  "Psoriasis": {
    emoji: "🔴",
    severity: "Moderate to Severe",
    icd10: "L40",
    description: "Psoriasis is an autoimmune condition that accelerates the skin cell life cycle. It causes cells to build up rapidly on the skin surface, forming scales and red patches that can be itchy and sometimes painful.",
    signs: ["Red patches of skin covered with thick, silvery scales", "Dry, cracked skin that may bleed", "Itching, burning, or soreness", "Thickened, pitted, or ridged nails", "Joint swelling and stiffness (psoriatic arthritis)"],
    care: [
      "Keep skin moisturized with thick creams or ointments",
      "Expose skin to small amounts of sunlight (as directed by a doctor)",
      "Avoid triggers: stress, infections, smoking, alcohol",
      "Take baths with colloidal oatmeal or Epsom salts",
      "Use gentle, fragrance-free products",
    ],
    when_to_consult: "Always consult a dermatologist for psoriasis management. Seek immediate care if psoriasis covers large areas or is accompanied by joint pain.",
    color: "#8b5cf6"
  },
  "Melanoma": {
    emoji: "⚠️",
    severity: "Serious — Seek Immediate Care",
    icd10: "C43",
    description: "Melanoma is the most serious type of skin cancer. It develops in the cells that produce melanin. If recognized and treated early, it is nearly always curable. However, if it spreads, it can be difficult to treat.",
    signs: ["A mole that changes in size, shape, or color", "A lesion with irregular, ragged, notched, or blurred border", "A mole with multiple colors or uneven pigmentation", "A mole larger than 6 mm in diameter", "A mole that is evolving, bleeding, or itching"],
    care: [
      "Apply broad-spectrum SPF 30+ sunscreen daily",
      "Wear protective clothing, hats, and UV-blocking sunglasses",
      "Avoid tanning beds and direct midday sun",
      "Perform regular self-skin exams",
      "Do NOT attempt home treatment — see a dermatologist immediately",
    ],
    when_to_consult: "⚠️ IMMEDIATELY. Melanoma requires urgent professional evaluation. Do not delay seeking medical attention if this result is returned.",
    color: "#ef4444"
  },
  "Ringworm": {
    emoji: "🔵",
    severity: "Mild (Contagious)",
    icd10: "B35",
    description: "Ringworm (tinea corporis) is a common fungal infection of the skin. Despite its name, it is not caused by a worm. It is highly contagious and causes a red, circular, itchy rash.",
    signs: ["Ring-shaped, red, scaly rash", "Itching", "Slightly raised, expanding rings", "Clearer or normal-looking skin in the middle of the ring", "On the scalp: round, scaly bald patches"],
    care: [
      "Keep the affected area clean and dry",
      "Apply antifungal cream (consult a pharmacist or doctor)",
      "Wash hands thoroughly after touching the affected area",
      "Do not share personal items (towels, combs, clothing)",
      "Wash bedding and clothing in hot water",
    ],
    when_to_consult: "If the rash spreads, worsens after a week of antifungal treatment, affects the scalp or nails, or if you have a weakened immune system.",
    color: "#10b981"
  },
  "Healthy Skin": {
    emoji: "✅",
    severity: "Normal",
    icd10: "Z13",
    description: "The AI model did not detect significant patterns associated with common skin conditions in the uploaded image. The skin appears to be within normal parameters based on visual analysis.",
    signs: ["Even skin tone", "Smooth texture", "No visible lesions or abnormal patches", "Normal coloring without unusual pigmentation"],
    care: [
      "Continue regular sun protection with SPF 30+ sunscreen",
      "Stay hydrated and maintain a balanced diet",
      "Moisturize daily with a product suitable for your skin type",
      "Perform regular self-examinations to monitor any changes",
      "Schedule periodic professional skin check-ups",
    ],
    when_to_consult: "Schedule a routine dermatology check-up at least once a year, or sooner if you notice any new or changing moles or skin changes.",
    color: "#10b981"
  }
};

document.addEventListener("DOMContentLoaded", async () => {
  if (!Auth.requireAuth()) return;

  const user = Auth.getUser();
  if (user) {
    document.querySelectorAll(".user-display-name").forEach(el => el.textContent = user.username);
    document.querySelectorAll(".user-avatar-initial").forEach(el => el.textContent = user.username.charAt(0).toUpperCase());
  }

  const resultStr = sessionStorage.getItem(CONFIG.RESULT_KEY);
  if (!resultStr) {
    showToast("No result found. Please run a new analysis.", "warning");
    setTimeout(() => window.location.href = "analyze.html", 2000);
    return;
  }

  const result = JSON.parse(resultStr);
  displayResult(result);
});

function displayResult(result) {
  const disease    = result.predicted_class || "Unknown";
  const confidence = result.confidence || 0;
  const pct        = (confidence * 100).toFixed(1);
  const info       = DISEASE_INFO[disease] || DISEASE_INFO["Healthy Skin"];
  const diseaseInfo = result.disease_info || {};

  // Header
  document.getElementById("result-disease").textContent     = disease;
  document.getElementById("result-date").textContent        = formatDate(result.created_at);
  document.getElementById("result-icd10").textContent       = diseaseInfo.icd10 || info.icd10 || "–";
  document.getElementById("result-severity").textContent    = diseaseInfo.severity || info.severity;
  document.getElementById("result-severity").className = `badge ${disease === "Melanoma" ? "badge-danger" : disease === "Healthy Skin" ? "badge-success" : "badge-warning"}`;

  // Image
  const imgEl = document.getElementById("result-image");
  if (result.image_path) {
    imgEl.src = `${CONFIG.API_BASE_URL}/${result.image_path}`;
    imgEl.onerror = () => imgEl.src = "";
  }

  // Confidence ring
  const conf = Math.round(parseFloat(pct));
  document.getElementById("conf-pct-text").textContent = pct + "%";
  const circle = document.getElementById("conf-circle");
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (conf / 100) * circumference;
  setTimeout(() => {
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = conf >= 80 ? "var(--success)" : conf >= 60 ? "var(--warning)" : "var(--danger)";
  }, 300);

  // Description
  const descEl = document.getElementById("disease-description");
  if (descEl) descEl.textContent = diseaseInfo.description || info.description;

  // Signs
  const signsEl = document.getElementById("disease-signs");
  if (signsEl) {
    const signs = info.signs;
    signsEl.innerHTML = signs.map(s => `<li style="display:flex; gap:8px; padding:8px 0; border-bottom:1px solid var(--border); font-size:.875rem; color:var(--text-secondary)"><span style="color:var(--primary); flex-shrink:0">•</span>${s}</li>`).join("");
  }

  // Care advice
  const careEl = document.getElementById("disease-care");
  if (careEl) {
    careEl.innerHTML = info.care.map(c => `<li style="display:flex; gap:10px; padding:8px 0; font-size:.875rem; color:var(--text-secondary)"><span style="color:var(--success); flex-shrink:0">✓</span>${c}</li>`).join("");
  }

  // When to consult
  const consultEl = document.getElementById("when-to-consult");
  if (consultEl) consultEl.textContent = info.when_to_consult;

  // Top 3 predictions
  const top3El = document.getElementById("top3-list");
  if (top3El && result.top3) {
    top3El.innerHTML = result.top3.map(p => {
      const pPct = (p.confidence * 100).toFixed(1);
      return `
        <div style="display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid var(--border)">
          <div style="flex:1; font-size:.875rem; font-weight:${p.class_name === disease ? "700" : "400"}; color:${p.class_name === disease ? "var(--primary)" : "var(--text-primary)"}">
            ${p.class_name === disease ? "★ " : ""}${p.class_name}
          </div>
          <div style="width:120px; height:6px; background:var(--bg-alt); border-radius:var(--radius-pill); overflow:hidden">
            <div style="height:100%; width:${pPct}%; background:${p.class_name === disease ? "var(--primary)" : "var(--border-dark)"}; border-radius:var(--radius-pill); transition: width .8s ease"></div>
          </div>
          <div style="width:50px; text-align:right; font-size:.875rem; font-weight:600; color:var(--text-primary)">${pPct}%</div>
        </div>`;
    }).join("");
  }

  // Save result ID for report download
  if (result.id) {
    document.getElementById("download-report-btn").setAttribute("data-id", result.id);
  }
}

async function downloadReport() {
  const btn = document.getElementById("download-report-btn");
  const predId = btn.getAttribute("data-id");
  if (!predId) {
    showToast("No prediction ID found. Please run a new analysis.", "warning");
    return;
  }

  setButtonLoading(btn, true, "Generating PDF…");
  try {
    const res = await API.get(`${CONFIG.ENDPOINTS.REPORT}/${predId}`);
    if (res.ok) {
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url; a.download = `DermaAI_Report_${predId}.pdf`;
      a.click(); URL.revokeObjectURL(url);
      showToast("Report downloaded successfully!", "success");
    } else {
      const data = await res.json();
      showToast(data.detail || "Failed to generate report.", "error");
    }
  } catch {
    showToast("Network error. Please try again.", "error");
  } finally {
    setButtonLoading(btn, false);
  }
}
