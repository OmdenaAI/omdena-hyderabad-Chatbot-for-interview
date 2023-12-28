import os
import streamlit.components.v1 as components
import streamlit as st

from typing import Tuple


# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
# (This is, of course, optional - there are innumerable ways to manage your
# release process.)
_RELEASE = True

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_component_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.

if not _RELEASE:
    _component_func = components.declare_component(
    "speech-to-text",
    url="http://localhost:3001",
)
else:
    # When we're distributing a production version of the component, we'll
    # replace the `url` param with `path`, and point it to the component's
    # build directory:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("speech-to-text", path=build_dir)
# Now the React interface only accepts an array of 1 or 2 elements.



# Edit arguments sent and result received from React component, so the initial input is converted to an array and returned value extracted from the component
def speech_to_text(key, callback = None, kwargs={}):
    component_value = _component_func()
    # print("after_component_func", component_value , type(component_value))
    if component_value is None:
        return ''
    if isinstance(component_value,dict) :
        if component_value["stopped"]:
            if(component_value['transcript'] == ''):
                return ''
            else:
                st.session_state[f"{key}_output"] = component_value["transcript"]
                if callback:
                    callback(**kwargs)
                    return component_value["transcript"]
                else:
                    return component_value["transcript"]
    else:
        return component_value