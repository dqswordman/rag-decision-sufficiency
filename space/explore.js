const questions = {
  "king-arthur": "Which actor plays King Arthur in Once Upon a Time?",
  ornans: "When was the creator of A Burial at Ornans born?",
};

const source = document.getElementById("case");
const details = document.getElementById("case-details");
const template = document.getElementById("decision-table");
const caseStatus = document.getElementById("case-status");
const resultsStatus = document.getElementById("results-status");

function appendRow(table, values) {
  const row = table.insertRow();
  const label = document.createElement("th");
  label.scope = "row";
  label.textContent = values[0];
  row.append(label);
  for (const value of values.slice(1)) row.insertCell().textContent = value;
}

function renderCase(cases) {
  const selected = cases[source.value];
  const proposal = selected.policies[1];
  const checked = selected.policies[2];
  const summary = document.createElement("p");
  summary.className = "case-note";
  summary.textContent = `${questions[source.value]} P1 estimated utility ${selected.proposal_utility.toFixed(3)} for ${proposal.context}. The check measured ${proposal.utility.toFixed(3)} and returned ${checked.context}.`;

  const result = template.content.cloneNode(true);
  result.querySelector("caption").textContent = `${selected.dataset} · ${selected.pool} · ${selected.model} · ${selected.query}`;
  const rows = result.querySelector("tbody");
  for (const policy of selected.policies) {
    appendRow(rows, [
      policy.name, policy.calls, policy.context, policy.utility.toFixed(3),
      policy.token_f1.toFixed(3), policy.exact_match,
    ]);
  }
  details.replaceChildren(summary, result);
}

function renderComparison(rows) {
  const result = template.content.cloneNode(true);
  const region = result.querySelector(".table-scroll");
  region.ariaLabel = "Mean normalized regret and target calls";
  result.querySelector("caption").textContent = "Paper comparison at the nine-call budget and after an optional check";
  const headers = result.querySelector("thead tr");
  headers.replaceChildren();
  for (const name of ["Policy", "Mean normalized regret", "Mean target calls"]) {
    const header = document.createElement("th");
    header.scope = "col";
    header.textContent = name;
    headers.append(header);
  }
  const body = result.querySelector("tbody");
  for (const row of rows) {
    appendRow(body, [row.name, row.mean_regret.toFixed(3), row.mean_calls.toFixed(2)]);
  }
  document.getElementById("results").replaceChildren(result);
}

async function loadArchive() {
  try {
    const response = await fetch("summary.json");
    if (!response.ok) throw new Error(`Archive returned HTTP ${response.status}`);
    const { cases, comparison } = await response.json();
    renderCase(cases);
    renderComparison(comparison);
    source.addEventListener("change", () => renderCase(cases));
    document.getElementById("decision-controls").hidden = false;
    caseStatus.hidden = true;
    resultsStatus.hidden = true;
  } catch {
    caseStatus.textContent = "The archived cases could not load. The paper companion contains both examples.";
    resultsStatus.textContent = "The comparison could not load. See the paper companion for the published results.";
  }
}

loadArchive();
