import React from 'react'
import ImageCard from './ImageCard'

export default function SimilarImages({ similarImagesData }) {
    return (
        <div className='mx-4'>
            <div className='d-flex justify-content-center fs-4 mb-2'>Similar Images Retrieved</div>
            <div className='d-flex flex-wrap justify-content-between'>
                {
                    similarImagesData['files'].map((imageFileName, index) => {
                        return (
                            <div className='mb-4'>
                                <ImageCard imageFileName={imageFileName}
                                    similarityValue={similarImagesData['similarityValues'][index]}
                                    imageClassification={similarImagesData['classifications'][index]}
                                />
                            </div>
                        )
                    })
                }
            </div>
        </div>
    )
}
