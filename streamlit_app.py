import streamlit as st
from PIL import Image
from utils import detect_signature, enhance_signature
from io import BytesIO
import io

st.title("📄 Signature Detection using YOLO")
st.write("Upload a document image and detect signatures using a YOLO model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Document", use_container_width=True)

    with st.spinner("Detecting signature..."):
        result_image, cropped_signatures = detect_signature(image)

    st.image(result_image, caption="Detected Signatures", use_container_width=True)

    if cropped_signatures:
        st.subheader("Cropped Signatures")

        for i, crop in enumerate(cropped_signatures):
            # Display cropped signature
            st.image(crop, caption=f"Cropped Signature #{i+1}", width=300)
            buf_jpg = crop.convert("RGB")
            st.download_button(
                label=f"📥 Download as JPEG #{i+1}",
                data=buf_jpg.tobytes("jpeg", "RGB"),
                file_name=f"signature_{i+1}.jpg",
                mime="image/jpeg"
            )

            # Display enhanced signature
            enhanced = enhance_signature(crop)
            st.image(enhanced, caption=f"Enhanced Signature #{i+1}", width=300)
            png_buf = io.BytesIO()
            enhanced.save(png_buf, format="PNG")
            st.download_button(
                label=f"✒️ Download Enhanced PNG #{i+1}",
                data=png_buf.getvalue(),
                file_name=f"enhanced_signature_{i+1}.png",
                mime="image/png"
            )

            # Optional: Divider between each signature pair
            st.markdown("---")



    else:
        st.warning("No signatures detected.")