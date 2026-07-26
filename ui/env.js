// Runtime config for the TENETDrill dashboard.
//
// This is a plain static site (no Vite build step), so there's no build-time
// VITE_API_URL substitution. This file is the equivalent mechanism: edit the
// one line below per deployment (or have your static-site build step
// template it) instead of rebuilding the app bundle.
//
// Local dev default points at `uvicorn api.app:app` running on 127.0.0.1:8000.
// For Render, set this to your deployed API's URL, e.g.
// "https://tenetdrill-api.onrender.com".
window.TENETDRILL_API_URL = window.TENETDRILL_API_URL || "http://127.0.0.1:8000";
