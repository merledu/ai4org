// Optional: small UX niceties
document.addEventListener('click', (e)=>{
  const pill = document.querySelector('.ai-pill');
  if(e.target === pill){
    pill.classList.add('pulse');
    setTimeout(()=>pill.classList.remove('pulse'), 700);
  }
});

// Arrays of headings and slogans
const headings = [
  "TRUST IN PRIVACY",
  "YOUR DATA, YOUR POWER",
  "PRIVACY REDEFINED",
  "SECURE BY DESIGN",
  "OWN YOUR INFORMATION"
];

const slogans = [
  "Your data, your control — never stored on cloud.",
  "No servers. No risks. Your data lives only with you.",
  "Privacy first. We don’t keep your data, you do.",
  "Your data is safe — offline, secure, and only yours.",
  "No tracking. No storing. 100% yours.",
  "Your digital life, protected. Nothing leaves your device.",
  "We never see your data — and never will.",
  "Trust in privacy: your information stays with you.",
  "Zero cloud. Zero compromise. Full privacy.",
  "Your data is safe, forever in your hands."
];

// Get the heading, slogan text, and video elements
// const securityHeading = document.getElementById('security-heading');
const securityHeading = document.getElementById('security-flow');

const sloganText = document.getElementById('slogan-text');
const securityVideo = document.getElementById('security-video');

// Set video playback speed (1.0 is normal, 1.5 is 50% faster, 0.5 is 50% slower, etc.)
securityVideo.playbackRate = 2; // Adjust this value as needed

// Function to change heading and slogan
function changeContent() {
  let headingIndex = 0;
  let sloganIndex = 0;

  setInterval(() => {
    // Fade out both heading and slogan
    securityHeading.style.opacity = 0;
    sloganText.style.opacity = 0;

    setTimeout(() => {
      // Update heading and slogan
      headingIndex = (headingIndex + 1) % headings.length;
      sloganIndex = (sloganIndex + 1) % slogans.length;
      securityHeading.textContent = headings[headingIndex];
      sloganText.textContent = slogans[sloganIndex];

      // Fade in both heading and slogan
      securityHeading.style.opacity = 1;
      sloganText.style.opacity = 1;
    }, 100); // Wait for fade out before changing text
  }, 800); // Change every 3 seconds
}

// Start the content cycle
changeContent();