import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import pickle
import numpy as np
from haversine import haversine
import pandas as pd
import helper
import random
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering

data = pickle.load(open('dataset.pkl', 'rb'))
model = pickle.load(open('svc.pkl', 'rb'))
lottie_coding = helper.load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_i04mzc5i.json")
lottie_coding1 = helper.load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_OX0Ts3.json")

##st.title('\tPathology Scheduling')
st.markdown("<h1 style='text-align: center; color: red;'>Pathology Specimen Collection Scheduling</h1>", unsafe_allow_html=True)
#project = st.sidebar.radio(menu_title = 'SELECT AN OPTION', options =  ['HOME', 'PREDICTION', 'SCHEDULING'])
project = option_menu(menu_title=None,
                      options=['HOME', 'PREDICTION', 'SCHEDULING'],
                      default_index=0,
                      orientation="horizontal",
                      icons=['house-fill', 'steam', 'telephone-plus-fill']
                      )

if project == 'PREDICTION':
    #st.sidebar.image('https://www.rabkindermpath.com/blog/admin/uploads/2020/rabkindx3.jpg')
    st.header('Prediction for Agent Arrival ')
    agent_id = st.selectbox('Select Agent ID', data['Agent ID'].unique())
    slot = st.selectbox('Select Booking Slot', ['06:00 to 21:00 (Home)' , '19:00 to 22:00 (working person)', '06:00 to 18:00 (Collect at work place)'])
    gender = st.radio('Select Gender', ['Female', 'Male'])
    storage = st.selectbox('Specimen Storage', ['Vacuum blood collection tube', 'Urine culture transport tube', 'Disposable plastic container'])
    distance = np.log(st.number_input('Distance Between Patient and Agent in Meters'))
    collection_time = st.number_input('Specimen collection Time in minutes')
    patient_from = st.number_input('PATIENT AVAILABLE FROM', min_value=1, value=20)
    if st.checkbox('Show Instruction 1'):
        st.text('In "PATIENT AVAILABLE FROM" input the time when patient is available for test\n'
                'Eg.: patient is available from 13(1PM) to 14(2PM)\n'
                'Note: value should be in 24-hour format')
    patient_to = st.number_input('PATIENT AVAILABLE TO', min_value=1, value=21)
    if st.checkbox('Show Instruction 2'):
        st.text('In "PATIENT AVAILABLE TO" input the time when patient is available upto for test\n'
                'Eg.: patient is available from 13(1PM) to 14(2PM)\n'
                'Note: value should be in 24-hour format')
    agent_before = st.number_input('PATIENT ARRIVED BEFORE', min_value=1, value=21)
    if st.checkbox('Show Instruction 3'):
        st.text('Eg.: agent will reach before 14(2PM)')

    if st.button('Predict Timing'):

        if slot == '06:00 to 18:00 (Collect at work place)':
            slot = 0
        elif slot == '06:00 to 21:00 (Home)':
            slot = 1
        elif slot == '19:00 to 22:00 (working person)':
            slot = 2

        if gender == 'Female':
            gender = 0
        elif gender == 'Male':
            gender = 1

        if storage == 'Disposable plastic container':
            storage = 0
        elif storage == 'Urine culture transport tube':
            storage = 1
        elif storage == 'Vacuum blood collection tube':
            storage = 2

        query = np.array([agent_id, slot, gender, storage, distance, collection_time, patient_from, patient_to, agent_before])
        query = query.reshape(1, 9)

        result = model.predict(query)

        if result == 24:
            st.success(f'Agent will reached within {24} minutes')
        elif result == 34:
            st.success(f'Agent will reached within {34} minutes')
        elif result == 39:
            st.success(f'Agent will reached within {39} minutes')
        elif result == 49:
            st.success(f'Agent will reached within {49} minutes')
        elif result == 54:
            st.success(f'Agent will reached within {54} minutes')
        else:
            st.success(f'Agent will reached within {64} minutes')
            st.write('Your Location is to far')
        ## gif
        st_lottie(lottie_coding, height=200)

if project == 'HOME':
    st.image('https://upload.wikimedia.org/wikipedia/commons/6/62/Latitude_and_Longitude_of_the_Earth.svg')
    if st.checkbox('HOW TO CALCULATE LATITUDES AND LONGITUDES ?'):
        st.subheader('STEP 1 : Open Maps\n')
        st.image('https://raw.githubusercontent.com/datasciritwik/test/main/step1.png')
        st.subheader('STEP 2 : Choose Agent Location then select coordinates\n')
        st.image('https://raw.githubusercontent.com/datasciritwik/test/main/Agent.png')
        st.write('Select and copy the coordinates appears on sidebar, eg: 17.431024, 78.373442')
        st.subheader('STEP 3 : Choose Patient Location then select coordinates\n')
        st.image('https://raw.githubusercontent.com/datasciritwik/test/main/patient.png')
        st.write('Select and copy the coordinates appears on sidebar, eg: 17.440018, 78.356908')
        st.subheader('STEP 4 : Input those coordinates into our formula')
    st.write("Click Above only you don't know how to calculate latitudes and longitudes else select below !")
    if st.checkbox('SHOW FORMULA'):
        st.subheader('HAVERSINE DISTANCE FORMULA')
        lat1 = round((st.number_input('AGENT LATITUDE')), 6)
        st.write('Enter up to minimum six decimal places')
        lon1 = round((st.number_input('AGENT LONGITUDE')), 6)
        st.write('Enter up to minimum six decimal places')
        lat2 = round((st.number_input('PATIENT LATITUDE')), 6)
        st.write('Enter up to minimum six decimal places')
        lon2 = round((st.number_input('PATIENT LONGITUDE')), 6)
        st.write('Enter up to minimum six decimal places')

        loc1 = (lat1, lon1)
        loc2 = (lat2, lon2)
        if st.checkbox('Show Coordinates'):
            st.write('Agent Location Coordinates', loc1)
            st.write('Patient Location Coordinates', loc2)
        if st.button('Calculate'):
            distance = int(haversine(loc1, loc2, unit='m'))
            st.success(f'Shortest Distance Between Agent and Patient is {distance} meters')
            ## gif
            st_lottie(lottie_coding1, height=200)

if project == 'SCHEDULING':
    df = pd.read_csv('final.csv')
    st.header('Agent Scheduling')
    df.rename(columns={'shortest distance Agent-Pathlab(m)': 'Distance Agent-Pathlab',  ##unit = meters
                       'shortest distance Patient-Pathlab(m)': 'Distance Patient-Pathlab',  ##unit = meters
                       'shortest distance Patient-Agent(m)': 'Distance Patient-Agent',  ##unit = meters
                       'Patient Availabilty': 'Patient Availability',  ##range format
                       'Test Booking Date': 'Booking Date',
                       'Test Booking Time HH:MM': 'Booking Time',
                       'Way Of Storage Of Sample': 'Specimen Storage',
                       ' Time For Sample Collection MM': 'Specimen collection Time',
                       'Time Agent-Pathlab sec': 'Agent-Pathlab sec',
                       'Agent Arrival Time (range) HH:MM': 'Agent Arrival Time',
                       'Exact Arrival Time MM': 'Exact Arrival Time'  ##output time
                       }, inplace=True)

    df['Diagnostic Centers'] = df['Diagnostic Centers'].apply(helper.name_change)
    df_subset = df[['Patient ID', 'pincode', 'Latitudes and Longitudes (Patient)', 'Age',
                    'Gender', 'Test name', 'Sample', 'Sample Collection Date', 'Specimen Storage', 'Time slot',
                    'Diagnostic Centers', 'Patient Availability', 'Agent ID', 'Distance Patient-Agent'
                    ]]
    df_subset['Distance Patient-Agent_log'] = np.log(df_subset['Distance Patient-Agent'])

    ## STEP 1 -----> (clustering entire dataset on the basis of distance column)

    df_subset = df_subset[df_subset['Distance Patient-Agent'] != 0]

    h_comp = AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='euclidean').fit(df_subset.iloc[:, 14:])
    df_subset['clusters'] = (pd.DataFrame(h_comp.labels_))
    df_subset.dropna(inplace=True)
    df_subset['clusters'] = df_subset['clusters'].astype('int64')
    df_subset = df_subset.sort_values(by='clusters')
    df_subset.drop(['Distance Patient-Agent', 'Distance Patient-Agent_log', 'clusters'], axis=1, inplace=True)

    ## STEP 2 ------->
    df_subset['Patient Availability from'] = df_subset['Patient Availability'].apply(
        lambda x: x.split('to')[0].split(':')[0])
    df_subset['Patient Availability to'] = df_subset['Patient Availability'].apply(
        lambda x: x.split('to')[1].split(':')[0])

    number_input = int(st.number_input('Number of Available Agent : ', min_value=1, value=1))
    agent_list = list(df_subset['Agent ID'])
    sampled_list = random.sample(agent_list, number_input)
    selected_df = pd.DataFrame()
    for i in sampled_list:
        data = df_subset[df_subset['Agent ID'] == i]
        selected_df = selected_df.append(data)
    # st.dataframe(selected_df['Agent ID'].unique())
    selected_df.drop_duplicates(inplace=True)
    # st.dataframe(selected_df.shape)
    clust = int(len(sampled_list))
    # st.write(clust)

    h_comp1 = AgglomerativeClustering(n_clusters=clust, linkage='complete', affinity='euclidean').fit(
        selected_df.iloc[:, 12:])
    selected_df['clusters'] = (pd.DataFrame(h_comp1.labels_))
    selected_df.dropna(inplace=True)
    selected_df['clusters'] = selected_df['clusters'].astype('int64')
    selected_df = selected_df.sort_values(by='clusters')

    if st.checkbox('SHOW DETAILS'):
        final = selected_df.drop(['Patient Availability from', 'Patient Availability to', 'clusters'], axis=1,
                                 inplace=True)
        st.dataframe(selected_df)

        ## 1st plot
        fig = px.bar(selected_df, x='Age', y='Specimen Storage', color='Gender', text_auto=True)
        st.title("Specimen Storage Types")
        st.plotly_chart(fig)
        ## 2nd plot
        fig = px.bar(selected_df, x='Age', y='Test name', color='Gender', text_auto=True)
        st.title("Test Name")
        st.plotly_chart(fig)

