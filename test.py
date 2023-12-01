import streamlit as st
# ... (your other imports)

# Initialize your camera or video processing outside the main loop
# ...


def main():
    st.title('Object Detection and Tracking')

    # Use the toggle button as you described



    col1, col2 = st.columns(2)

    with col1:
        cam1_on = st.toggle('Activate Camera 1')
        if cam1_on:
            st.write('Camera 1 activated!')
            # Code to display Camera 1 feed
            # ...

    with col2:
        cam2_on = st.toggle('Activate Camera 2')
        if cam2_on:
            st.write('Camera 2 activated!')
            # Code to display Camera 2 feed
            # ...

    # Rest of your code for object detection and tracking
    # ...

if __name__ == "__main__":
    main()




"""
# Initialize session state for toggles if not already present
if 'cam1_on' not in st.session_state:
    st.session_state['cam1_on'] = False
if 'cam2_on' not in st.session_state:
    st.session_state['cam2_on'] = False

# Layout for toggle buttons
col1, col2 = st.columns(2)

with col1:
    cam1_on = st.toggle('Activate Camera 1')
    if cam1_on:
        st.session_state['cam1_on'] = True
        st.write('Camera 1 activated!')
        # Code to display Camera 1 feed and plots
       # process_camera_feed(0, frame_placeholder1, plot_placeholder1)
    else:
        st.session_state['cam1_on'] = False
        st.write('Camera 1 deactivated.')

with col2:
    cam2_on = st.toggle('Activate Camera 2')
    if cam2_on:
        st.session_state['cam2_on'] = True
        st.write('Camera 2 activated!')
        # Code to display Camera 2 feed and plots
        #process_camera_feed(1, frame_placeholder2, plot_placeholder2)
    else:
        st.session_state['cam2_on'] = False
        st.write('Camera 2 deactivated.')"""
