import React from 'react'
import NavBar from './NavBar'
import Form from './Form'

export default function Home() {
    return (
        <div>
            <NavBar />
            <div className='d-flex flex-row justify-content-center fs-2 fw-semibold'>
                CONTENT BASED IMAGE RETRIEVAL
            </div>

            <Form />
        </div>
    )
}
