import streamlit as st
import time


def duration_timer(duration):
    with st.empty():
        for s in range(duration):
            second = duration - s if duration - s >= 10 else "0" + str(duration - s)
            st.write(f"⏳ 00:{second} seconds have passed")
            time.sleep(1)
        st.write(f"✔️ {duration}s music has recorded")


duration_timer(15)
