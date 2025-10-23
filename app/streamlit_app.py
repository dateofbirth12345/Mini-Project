from __future__ import annotations

import os
import json
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import folium
from streamlit_folium import st_folium
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.inference import predict_on_images

st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Navigation Bar




st.set_page_config(
    page_title="CleanSea Vision",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ocean-themed custom styling
st.markdown(
   """ 
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {padding-top: 0rem;}
    .stApp {background: linear-gradient(135deg, #0F4C75 0%, #80A1BA 25%, #91C4C3 50%, #B4DEBD 75%, #FFF7DD 100%);}
    
    
    
    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #0F4C75 0%, #3282B8 25%, #2E86AB 50%, #1B4F72 75%, #0F4C75 100%);
        color: beige;
        padding: 6rem 2rem;
        text-align: center;
        border-radius: 20px;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        min-height: 60vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="wave" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse"><path d="M0,50 Q25,25 50,50 T100,50 L100,100 L0,100 Z" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23wave)"/></svg>');
        opacity: 0.3;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 800px;
    }
    
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.03em;
        line-height: 0.95;
        text-transform: uppercase;
    }
    
    .hero p {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        margin-bottom: 3rem;
        opacity: 0.95;
        font-weight: 400;
        line-height: 1.6;
        letter-spacing: 0.01em;
    }
    
    .cta-button {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(45deg, #F4A261, #E76F51);
        color: white;
        padding: 1.2rem 3rem;
        border: none;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        box-shadow: 0 6px 20px rgba(244, 162, 97, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(244, 162, 97, 0.5);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .card-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(45deg, #0F4C75, #3282B8);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0F4C75;
        margin: 0;
    }
    
    /* Stats */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F4C75;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #2C5530;
        font-weight: 500;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #0F4C75 0%, #1B4F72 50%, #2E86AB 100%);
        color: white;
        padding: 3rem 2rem;
        text-align: center;
        margin: 3rem -1rem -1rem -1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .nav-menu {display: none;}
        .hero h1 {font-size: 2.5rem;}
        .hero p {font-size: 1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Hero Section
st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <h1>CLEANSEA VISION</h1>
            <p>Protecting our oceans through technology and community action. Join thousands of citizens worldwide in detecting and reporting marine debris to create a cleaner, healthier planet.</p> 
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Center DONATE button that navigates to donation tab
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("DONATE", type="primary", use_container_width=True, key="hero_donate"):
        st.session_state.active_tab = "💙 Donate"
        st.rerun()


# Stats Section
st.markdown(
    """
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">2,847</div>
            <div class="stat-label">Debris Reports</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">156</div>
            <div class="stat-label">Cleanup Events</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">89%</div>
            <div class="stat-label">Detection Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">47</div>
            <div class="stat-label">Countries</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
def draw_boxes(image_path: str, boxes: List[dict]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        color = _color_for_class(int(b["cls"]))
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{b['label']} {b['conf']:.2f}"
        tw, th = draw.textlength(label, font=font), 16
        # label background
        draw.rectangle([(x1, max(0, y1 - th - 4)), (x1 + int(tw) + 8, y1)], fill=(0, 0, 0, 160))
        draw.text((x1 + 4, y1 - th - 2), label, fill=(255, 255, 255), font=font)
    return img
def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
    palette = [
        (39, 76, 119),  # deep blue
        (51, 92, 103),  # teal
        (237, 125, 49), # orange
        (120, 94, 240), # purple
        (35, 166, 213), # sky
        (23, 190, 187), # cyan
    ]
    return palette[cls_id % len(palette)]
st.markdown("---")

# Main Content Tabs
# Initialize session state for tab navigation
st.markdown("---")

# Main Content Navigation using buttons instead of tabs
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📸 Help Cleanup"

# Tab-like navigation buttons
st.markdown("### Navigation")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📸 Help Cleanup", use_container_width=True, type="primary" if st.session_state.active_tab == "📸 Help Cleanup" else "secondary"):
        st.session_state.active_tab = "📸 Help Cleanup"

with col2:
    if st.button("💙 Donate", use_container_width=True, type="primary" if st.session_state.active_tab == "💙 Donate" else "secondary"):
        st.session_state.active_tab = "💙 Donate"

with col3:
    if st.button("📊 Updates", use_container_width=True, type="primary" if st.session_state.active_tab == "📊 Updates" else "secondary"):
        st.session_state.active_tab = "📊 Updates"

with col4:
    if st.button("🌍 Waterbodies", use_container_width=True, type="primary" if st.session_state.active_tab == "🌍 Polluted Waterbodies" else "secondary"):
        st.session_state.active_tab = "🌍 Polluted Waterbodies"

with col5:
    if st.button("🤝 Join Team", use_container_width=True, type="primary" if st.session_state.active_tab == "🤝 Join Team" else "secondary"):
        st.session_state.active_tab = "🤝 Join Team"

st.markdown("---")

# Conditional rendering based on active tab
if st.session_state.active_tab == "📸 Help Cleanup":
    st.markdown("### 📸 Upload Photos to Detect Marine Debris")
    st.markdown("Help us identify and locate marine debris by uploading photos from drones, boats, or shorelines. Our AI will analyze the images in real-time.")
    
   
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📁 Upload images (JPG/PNG)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            help="Upload clear photos of suspected marine debris. Multiple files supported."
        )
        
        if uploaded_files:
            tmp_dir = "runs/ui_uploads"
            os.makedirs(tmp_dir, exist_ok=True)
            image_paths: List[str] = []
            for f in uploaded_files:
                img = Image.open(f).convert("RGB")
                out_path = os.path.join(tmp_dir, f.name)
                img.save(out_path)
                image_paths.append(out_path)

            with st.spinner("🔍 Detecting debris with AI..."):
                results = predict_on_images(image_paths)

            for item in results:
                st.markdown(f"**📷 {os.path.basename(item['image_path'])}**")
                if len(item["boxes"]) == 0:
                    st.info("✅ No debris detected in this image.")
                    st.image(item["image_path"], use_container_width=True)
                else:
                    vis = draw_boxes(item["image_path"], item["boxes"])
                    st.image(vis, use_container_width=True)
                    st.markdown(
                        " ".join(
                            [
                                f"<span class='metric-pill'>{b['label']} · {b['conf']:.2f}</span>"
                                for b in item["boxes"][:6]
                            ]
                        ),
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div class='caption'>📍 GPS Location: {item['gps'] if item['gps'] else 'Not available (Enable location services)'}</div>",
                    unsafe_allow_html=True
                )
                st.divider()
    
    with col2:
        st.markdown("### ⚙️ Detection Settings")
        conf = st.slider("🎯 Detection confidence", 0.05, 0.90, 0.25, 0.01, help="Lower = more detections, Higher = more accurate")
        iou = st.slider("🔲 NMS IoU threshold", 0.10, 0.90, 0.45, 0.01, help="Controls overlapping box removal")
        
        st.info("""
        **Tips:**
        - Lower confidence → more detections
        - Higher confidence → fewer, stronger detections
        - Adjust based on your image quality
        """)
        
        if uploaded_files:
            gps_points = [it["gps"] for it in results if it["gps"] is not None]
            st.markdown("### 🗺️ Detection Map")
            if len(gps_points) > 0:
                first_lat, first_lon = gps_points[0]  # type: ignore[index]
                m = folium.Map(location=[first_lat, first_lon], tiles="OpenStreetMap", zoom_start=11)
                for it in results:
                    if it["gps"] is not None:
                        lat, lon = it["gps"]
                        folium.Marker(
                            location=[lat, lon],
                            popup=f"📷 {os.path.basename(it['image_path'])}",
                            icon=folium.Icon(color="red", icon="camera", prefix='fa'),
                        ).add_to(m)
                st_folium(m, width=None, height=400)
            else:
                st.warning("📍 No GPS data found in uploaded photos. Enable camera location services to see pins on the map.")

elif st.session_state.active_tab == "💙 Donate":
    st.markdown("### 💙 Support Our Ocean Conservation Mission")
    st.markdown("Your donation helps us maintain AI detection technology, organize global cleanup events, and protect marine ecosystems worldwide. Every contribution makes a difference!")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 💳 Make a Donation")
        
        donation_amount = st.selectbox(
            "Select Amount",
            ["₹100", "₹500", "₹1000", "₹2500", "₹5000", "Custom Amount"],
            help="Choose a preset amount or enter custom"
        )
        
        custom_amount = None
        if donation_amount == "Custom Amount":
            custom_amount = st.number_input("Enter Amount (₹)", min_value=10, max_value=1000000, value=1000)
            amount = custom_amount
        else:
            amount = int(donation_amount.replace("₹", ""))
        
        st.markdown("#### 👤 Donor Information")
        donor_name = st.text_input("Full Name *", placeholder="Enter your full name")
        donor_email = st.text_input("Email *", placeholder="your.email@example.com")
        donor_phone = st.text_input("Phone (Optional)", placeholder="+91 98765 43210")
        anonymous = st.checkbox("🕶️ Donate anonymously")

    with col2:
        st.markdown("#### 🏦 Payment Methods")
        
        payment_method = st.radio(
            "Choose Payment Method",
            ["💳 UPI", "💎 Credit/Debit Card", "🏦 Net Banking", "📱 Digital Wallet"],
            help="Select your preferred payment method"
        )
        
        if payment_method == "💳 UPI":
            st.markdown("##### Scan QR or Use UPI ID")
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: var(--light-bg); border-radius: 12px; margin: 1rem 0; border: 2px solid var(--border-color);">
                <p style="font-weight: 600; color: var(--text-dark);"><strong>UPI ID:</strong> cleanseavision@paytm</p>
                <div style="width: 200px; height: 200px; background: white; margin: 1rem auto; border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: var(--shadow-sm); border: 2px dashed var(--border-color);">
                    <span style="color: var(--text-light); font-size: 3rem;">📱</span>
                </div>
                <p style="color: var(--text-light); font-size: 0.9rem;">Scan with any UPI app</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif payment_method == "💎 Credit/Debit Card":
            st.markdown("##### Card Payment Details")
            card_number = st.text_input("Card Number *", placeholder="1234 5678 9012 3456")
            col_exp, col_cvv = st.columns(2)
            with col_exp:
                expiry = st.text_input("Expiry (MM/YY) *", placeholder="12/25")
            with col_cvv:
                cvv = st.text_input("CVV *", placeholder="123", max_chars=4, type="password")
            card_name = st.text_input("Name on Card *", placeholder="JOHN DOE")
            
        elif payment_method == "🏦 Net Banking":
            st.markdown("##### Select Your Bank")
            bank = st.selectbox("Bank Name *", [
                "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", 
                "Kotak Mahindra Bank", "Punjab National Bank", "Bank of Baroda", "Canara Bank"
            ])
            
        elif payment_method == "📱 Digital Wallet":
            st.markdown("##### Select Wallet Provider")
            wallet = st.selectbox("Wallet *", [
                "Paytm", "PhonePe", "Google Pay", "Amazon Pay", "Mobikwik", "Freecharge"
            ])

    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💙 Complete Donation", type="primary", use_container_width=True):
            if not donor_name or not donor_email:
                st.error("⚠️ Please fill in your name and email address.")
            else:
                with st.spinner("Processing your donation..."):
                    import time
                    time.sleep(2)
                
                st.success(f"🎉 Thank you for your generous donation of ₹{amount:,}! Your contribution helps protect our oceans.")
                
                donation_record = {
                    "amount": f"₹{amount:,}",
                    "donor_name": "Anonymous Donor" if anonymous else donor_name,
                    "email": donor_email,
                    "payment_method": payment_method,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "receipt_id": f"CSR-{random.randint(100000, 999999)}"
                }
                
                st.markdown("### 📧 Donation Receipt")
                st.json(donation_record)
                st.info("📧 A detailed receipt has been sent to your email address. Thank you for supporting ocean conservation! 🌊")

elif st.session_state.active_tab == "📊 Updates":
    st.markdown("### 📊 Latest Cleanup Updates & Success Stories")
    
    # Search and filter bar
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        st.markdown(
            """
            <div class="search-container">
                <span class="search-icon">🔍</span>
                <input type="text" class="search-box" placeholder="Search updates by location, date, or keyword..." />
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_filter:
        impact_filter = st.selectbox("Filter Impact", ["All", "High", "Medium", "Low"])
    
    updates_data = [
        {
            "date": "2024-01-15",
            "location": "Great Pacific Garbage Patch",
            "description": "Successfully removed 2.3 tons of plastic debris using drone detection and coordinated cleanup teams",
            "participants": 45,
            "impact": "High",
            "icon": "🌊"
        },
        {
            "date": "2024-01-12",
            "location": "French Riviera, Mediterranean",
            "description": "Community-led beach cleanup collected 1.8 tons of waste and microplastics",
            "participants": 120,
            "impact": "Medium",
            "icon": "🏖️"
        },
        {
            "date": "2024-01-10",
            "location": "Caribbean Sea",
            "description": "Professional divers removed 500kg of abandoned fishing gear and plastic waste",
            "participants": 25,
            "impact": "High",
            "icon": "🤿"
        },
        {
            "date": "2024-01-08",
            "location": "North Sea, Netherlands",
            "description": "AI-powered detection helped locate and remove 800kg of microplastics from beaches",
            "participants": 80,
            "impact": "Medium",
            "icon": "🔬"
        }
    ]
    
    for update in updates_data:
        with st.expander(f"{update['icon']} {update['date']} - {update['location']}", expanded=False):
            st.markdown(f"**Description:** {update['description']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👥 Participants", update['participants'])
            with col2:
                st.metric("📊 Impact Level", update['impact'])
            with col3:
                st.metric("Status", "✅ Completed")

elif st.session_state.active_tab == "🌍 Polluted Waterbodies":
    st.markdown("### 🌍 Global Polluted Waterbodies Database")
    st.markdown("Comprehensive data sourced from international environmental agencies, satellite imagery, and citizen science reports.")
    
    # Enhanced search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            """
            <div class="search-container">
                <span class="search-icon">🔍</span>
                <input type="text" class="search-box" placeholder="Search by waterbody name, country, or region..." />
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        pollution_filter = st.selectbox("🚨 Pollution Level", ["All", "Critical", "High", "Medium", "Low"])
    with col3:
        region_filter = st.selectbox("🌏 Region", ["All", "Asia", "Europe", "Americas", "Africa", "Oceania"])
    
    waterbodies_data = {
        "Rank": list(range(1, 26)),
        "Waterbody": [
            "Great Pacific Garbage Patch", "Mediterranean Sea", "Caribbean Sea",
            "North Sea", "Baltic Sea", "Gulf of Mexico", "South China Sea",
            "Indian Ocean (Arabian Sea)", "Atlantic Ocean (Sargasso Sea)", "Pacific Ocean (North)",
            "Yangtze River", "Ganges River", "Nile River", "Amazon River",
            "Mississippi River", "Thames River", "Seine River", "Rhine River",
            "Danube River", "Volga River", "Yellow River", "Mekong River",
            "Congo River", "Niger River", "Pearl River"
        ],
        "Pollution Level": ["Critical"] * 7 + ["High"] * 10 + ["Medium"] * 8,
        "Debris Count (tons)": [random.randint(5000, 50000) for _ in range(25)],
        "Region": ["Pacific", "Europe", "Americas", "Europe", "Europe", "Americas", "Asia",
                  "Asia", "Atlantic", "Pacific", "Asia", "Asia", "Africa", "Americas",
                  "Americas", "Europe", "Europe", "Europe", "Europe", "Europe",
                  "Asia", "Asia", "Africa", "Africa", "Asia"],
        "Last Updated": [(datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d') for _ in range(25)]
    }
    
    df = pd.DataFrame(waterbodies_data)
    
    if pollution_filter != "All":
        df = df[df['Pollution Level'] == pollution_filter]
    
    if region_filter != "All":
        df = df[df['Region'] == region_filter]
    
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("🏆 Rank", width="small"),
            "Waterbody": st.column_config.TextColumn("🌊 Waterbody", width="large"),
            "Pollution Level": st.column_config.TextColumn("⚠️ Pollution", width="small"),
            "Debris Count (tons)": st.column_config.NumberColumn("📦 Debris (tons)", format="%d"),
            "Region": st.column_config.TextColumn("🌏 Region", width="small"),
            "Last Updated": st.column_config.TextColumn("📅 Updated", width="medium")
        },
        hide_index=True,
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Click column headers to sort. Use filters above to narrow results.")

elif st.session_state.active_tab == "🤝 Join Team":
    st.markdown("### 🤝 Join Our Global Cleanup Team")
    st.markdown("Make a direct impact by volunteering, organizing events, or spreading awareness about marine conservation.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon">👥</div>
                <h3 class="card-title">Volunteer Opportunities</h3>
            </div>
            <p>Join local and international cleanup events to make a tangible difference.</p>
            <ul>
                <li>Beach and coastal cleanups</li>
                <li>Underwater diving expeditions</li>
                <li>River and lake restoration</li>
                <li>Educational workshops & seminars</li>
                <li>Data collection & AI training</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon">📢</div>
                <h3 class="card-title">Spread Awareness</h3>
            </div>
            <p>Help us reach more people and amplify our impact globally.</p>
            <ul>
                <li>Share on social media platforms</li>
                <li>Organize community awareness events</li>
                <li>Conduct school & university talks</li>
                <li>Partner with local NGOs</li>
                <li>Create content & blogs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 Volunteer Registration")
    
    col1, col2 = st.columns(2)
    with col1:
        volunteer_name = st.text_input("Full Name *", placeholder="Enter your name")
        volunteer_email = st.text_input("Email *", placeholder="your.email@example.com")
        volunteer_phone = st.text_input("Phone *", placeholder="+91 98765 43210")
    
    with col2:
        volunteer_location = st.text_input("Location *", placeholder="City, Country")
        volunteer_interest = st.multiselect("Areas of Interest", [
            "Beach Cleanups", "Underwater Cleanups", "River Cleanups",
            "Event Organization", "Education & Awareness", "Photography",
            "Social Media", "Data Analysis", "Fundraising"
        ])
        volunteer_availability = st.selectbox("Availability", [
            "Weekends only", "Weekdays only", "Flexible", "Monthly events", "One-time event"
        ])
    
    volunteer_message = st.text_area("Tell us about yourself (optional)", placeholder="Share your experience, motivation, or any questions...")
    
    if st.button("🤝 Register as Volunteer", type="primary", use_container_width=True):
        if not volunteer_name or not volunteer_email or not volunteer_phone or not volunteer_location:
            st.error("⚠️ Please fill in all required fields marked with *")
        else:
            st.success("🎉 Thank you for registering! We'll contact you soon with volunteer opportunities.")
            st.balloons()
    
    st.markdown("---")
    st.markdown("### 📞 Contact Information")
    st.markdown("""
    <div class="card">
        <p><strong>📧 Email:</strong> volunteer@cleanseavision.org</p>
        <p><strong>📱 Phone:</strong> +1 (555) 123-4567</p>
        <p><strong>🌐 Website:</strong> www.cleanseavision.org</p>
        <p><strong>📱 Social Media:</strong> @CleanSeaVision on Instagram, Twitter, Facebook</p>
    </div>
    """, unsafe_allow_html=True)