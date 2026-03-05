/* ───────────────────────────────────────────
   NeuroPulse — main.js
─────────────────────────────────────────── */

// ── Chart.js global defaults ───────────────
function chartDefaults() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'DM Sans', size: 12 }, boxWidth: 12 }
      },
      tooltip: {
        backgroundColor: '#1a2235',
        borderColor: '#1e2d45',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        padding: 12,
      }
    },
  };
}

function yAxis() {
  return {
    grid:   { color: 'rgba(30,45,69,0.6)' },
    ticks:  { color: '#475569', font: { family: 'Space Mono', size: 11 } },
    border: { color: '#1e2d45' },
  };
}

function xAxis() {
  return {
    grid:   { color: 'rgba(30,45,69,0.4)' },
    ticks:  { color: '#475569', font: { family: 'Space Mono', size: 10 } },
    border: { color: '#1e2d45' },
  };
}

// ── Auto-dismiss alerts ────────────────────
document.querySelectorAll('.alert').forEach(el => {
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transition = 'opacity 0.4s';
    setTimeout(() => el.remove(), 400);
  }, 4000);
});

// ── Fade-in on scroll ──────────────────────
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity = '1';
      e.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.card, .stat-card').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(16px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(el);
});

// ── Active nav highlight ───────────────────
const currentPath = window.location.pathname;
document.querySelectorAll('.nav-link').forEach(link => {
  if (link.getAttribute('href') === currentPath) {
    link.classList.add('active');
  }
});
