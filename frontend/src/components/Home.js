import React, { useState } from 'react'
import NavBar from './NavBar'
import Form from './Form'
import SimilarImages from './SimilarImages'

export default function Home() {
    const [similarImagesData, setSimilarImagesData] = useState({})

    return (
        <div>
            <NavBar />
            <div className='d-flex flex-row justify-content-center fs-2 fw-semibold'>
                CONTENT BASED IMAGE RETRIEVAL
            </div>

            <Form setSimilarImagesData={setSimilarImagesData} />

            {
                similarImagesData.hasOwnProperty('files') && (<SimilarImages similarImagesData={similarImagesData} />)
            }


        </div>
    )
}
