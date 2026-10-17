import os
import streamlit as st
from PIL import Image
from utils.predict import predict_image

# Page Configuration - Clean title without emojis
st.set_page_config(
    page_title="Currency Note Recognition Assistant",
    layout="wide"
)

# Custom Styling for clean, modern enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .result-card-valid {
        background-color: #F0FDF4;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 6px solid #10B981;
        margin-top: 1rem;
        border: 1px solid #DCFCE7;
    }
    .result-card-invalid {
        background-color: #FEF2F2;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 6px solid #EF4444;
        margin-top: 1rem;
        border: 1px solid #FEE2E2;
    }
    .country-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .country-pkr {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
    .country-afn {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .denom-badge {
        font-size: 2rem;
        font-weight: 800;
        color: #047857;
        margin-top: 0.3rem;
    }
    .invalid-badge {
        font-size: 1.8rem;
        font-weight: 800;
        color: #DC2626;
        margin-top: 0.3rem;
    }
    .stage-item {
        margin: 6px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Information
with st.sidebar:
    st.header("About the Project")
    st.write(
        "This assistant uses a deep fine-tuned **ResNet-50** neural network "
        "to automatically detect and classify currency banknotes across multiple stages."
    )

    st.subheader("Decision Pipeline")
    st.markdown("""
    1. **Banknote Verification:** Distinguishes genuine banknotes from non-currency objects (vehicles, furniture, surfaces).
    2. **Country Identification:** Confirms whether the currency is Afghan Afghani (AFN) or Pakistani Rupee (PKR).
    3. **Denomination Recognition:** Pinpoints the exact note value.
    """)

    st.subheader("Supported Denominations:")
    st.markdown("""
    * **Afghan Afghani (AFN):** `10, 20, 50, 100, 500, 1000`
    * **Pakistani Rupee (PKR):** `10, 20, 50, 100, 500, 1000, 5000`
    """)

    st.divider()
    st.markdown("**Developed by Tuhid Khan**")
    st.markdown("[GitHub Profile](https://github.com/TuhidKhan) &bull; [LinkedIn Profile](https://www.linkedin.com/in/tuhid-khan/)")
    st.caption("Deep Learning Computer Vision System")

# Main Header
st.markdown('<div class="main-header">Currency Note Recognition Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image of an Afghan Afghani or Pakistani Rupee banknote to classify its country and denomination. Non-banknote images will be automatically rejected.</div>', unsafe_allow_html=True)

# Check model files
model_exists = os.path.exists("models/model.pth") and os.path.exists("models/class_names.json")

if not model_exists:
    st.warning("Model weights not found in models/model.pth. Please complete training first.")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload Banknote Image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    image = Image.open(uploaded_file)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Recognition Analysis")

        if not model_exists:
            st.error("Cannot run prediction: Model weights are not available yet.")
        else:
            with st.spinner("Analyzing image features..."):
                result = predict_image(image)

            is_valid = result.get("is_valid", False)
            conf = result.get("confidence", 0.0)

            if not is_valid:
                st.markdown(f"""
                <div class="result-card-invalid">
                    <p style="margin: 0; color: #991B1B; font-weight: 600; font-size: 0.85rem;">STAGE 1: VERIFICATION STATUS</p>
                    <div class="invalid-badge">Unrecognized Object</div>
                    <p style="margin: 0.6rem 0 0 0; color: #7F1D1D; font-size: 0.95rem;">
                        {result['message']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.write(f"**Peak Pattern Match:** `{result['top_predictions'][0]['class'].replace('_', ' ')}` ({conf}%)")
                st.progress(min(1.0, conf / 100.0))

            else:
                country_code = result["country_code"]
                country_name = result["country_name"]
                denom = result["denomination"]
                country_class = "country-pkr" if country_code == "PKR" else "country-afn"

                st.markdown(f"""
                <div class="result-card-valid">
                    <span class="country-tag {country_class}">{country_code} &bull; {country_name.upper()}</span>
                    <p style="margin: 0.8rem 0 0 0; color: #065F46; font-size: 0.85rem; font-weight: 600;">CONFIRMED DENOMINATION</p>
                    <div class="denom-badge">{denom} {country_code}</div>
                    <p style="margin: 0.4rem 0 0 0; color: #047857; font-size: 0.95rem;">
                        Valid {country_name} banknote verified with high certainty.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.write(f"**Confidence Level:** `{conf}%` (Country Aggregate: `{result.get('country_confidence', 0)}%`)")
                st.progress(min(1.0, conf / 100.0))

            # Breakdown
            st.write("---")
            st.write("**Top Class Probabilities:**")
            for pred in result["top_predictions"]:
                c_name, c_denom = pred["class"].split("_")
                st.write(f"- **{c_name} {c_denom}**: `{pred['confidence']}%`")

else:
    st.info("Upload an image above to begin automatic detection.")
