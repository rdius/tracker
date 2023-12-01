import streamlit as st
from trackme import track
import streamlit.components.v1 as components
import datetime

DEFAULT_MAP_SPECS = dict(height=450, width=600)  # in px
APP_TITLE = 'SURVEILLANCE'
LOGO = ""# "https://technatium.com/wp-content/uploads/2022/09/cropped-TECHNATIUM-LOGO-SANSFOND.png"


def add_code_logo(logowidth: str = "500px"):
    CODE_LOGO = "https://technatium.com/wp-content/uploads/2022/09/cropped-TECHNATIUM-LOGO-SANSFOND.png" #"https://www.pngitem.com/pimgs/m/77-779399_transparent-homework-icon-png-blue-code-icon-png.png"
    st.markdown(
        f'<img src="{CODE_LOGO}" width="{logowidth}"/>',
        unsafe_allow_html=True,
    )

#@st.cache(suppress_st_warning=True)
def add_title(uselogo: bool = True, logowidth: str='500px'):
    col1, col2, col3, col4= st.columns(4)
    with col1:
        st.write(f'# {APP_TITLE}')
    with col4:
        if uselogo:
            st.markdown(
                f'<img src="{LOGO}" width="{logowidth}"/>',
                unsafe_allow_html=True,
            )


def my_app(wide_layout:bool=False): #
    if wide_layout:
        #layout = 'wide'
        st.set_page_config(layout="wide", page_icon=LOGO, page_title=APP_TITLE)
        #st.set_page_config(page_icon=LOGO, page_title=APP_TITLE)
        add_title(uselogo=True, logowidth='250px')



    map_specs = DEFAULT_MAP_SPECS


    #available_tiles = ["Planes & Cars", "Trees & Buildings", "Luxemb Airport", "Luxemb Airport - Static"]
    available_tiles = ["Tracker"]

    DEFAULT_TILES = "Tracker"
    tiles = dict()
    with st.sidebar:
        option = st.radio("Choose a site to monitor",
                          options= available_tiles)
   


    if option == "Tracker":
        track()
